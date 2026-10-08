// PhysX 5 backend for wi::physics. Selected at build time with -DWICKED_PHYSICS_BACKEND=PhysX; the Jolt backend
//	(wiPhysics_Jolt.cpp) is the default and the two are never compiled together, they define the same symbols.
//
// Why a second backend: the engine is being evaluated as a quadruped robot simulator, and the question is how the
//	solver type shows in contact and joint behaviour. Jolt is a PGS impulse solver over a body-and-constraint graph;
//	PhysX articulations are reduced coordinate with a TGS solver. Running the same robot, the same controller and the
//	same measurement client over both (and over MuJoCo through the same interface) is what makes the difference
//	measurable rather than argued.
//
// Notes that matter when reading this file:
//	- Vectors pass straight through, component for component, exactly as the Jolt backend does. Both engines are
//	  handed the numbers the engine holds; there is no handedness or axis conversion anywhere in a backend.
//	- Reduced coordinates make several things the Jolt backend has to approximate native here: joint armature, dry
//	  joint friction, and the joint zero. Those approximations are what this backend exists to measure against.
//	- Joint position and velocity must be read from the same PxArticulationCache. Reading the angle from the link
//	  poses and the velocity from the link velocities makes the two disagree wherever the solver corrects position,
//	  which is measurable and which the Jolt backend does suffer from.
//	- The engine calls back into wi::physics during static destruction (applications commonly keep the Application
//	  object as a global). Anything with a destructor that the shutdown path can reach is leaked on purpose.

#include "wiPhysics.h"

#include "wiScene.h"
#include "wiProfiler.h"
#include "wiBacklog.h"
#include "wiJobSystem.h"
#include "wiRenderer.h"
#include "wiTimer.h"
#include "wiSpinLock.h"

#include <PxPhysicsAPI.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <string>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

using namespace wi::ecs;
using namespace wi::scene;
using namespace physx;

namespace wi::physics
{
	namespace
	{
		bool ENABLED = true;
		bool SIMULATION_ENABLED = true;
		bool INTERPOLATION = true;
		bool DEBUGDRAW_ENABLED = false;
		float CONSTRAINT_DEBUGSIZE = 1;
		int ACCURACY = 4;
		float TIMESTEP = 1.0f / 60.0f;
		bool ASYNC_SIMULATION = false;
		float CHARACTER_COLLISION_TOLERANCE = 0.05f;
		float DEBUG_MAX_DRAW_DISTANCE = 500.0f;
		// PhysX asks for these the other way around from Jolt (position first). These are the SDK defaults.
		int SOLVER_VELOCITY_ITERATIONS = 1;
		int SOLVER_POSITION_ITERATIONS = 4;

		static constexpr uint32_t dispatchGroupSize = 256u;

		// Features that this backend does not implement. Each one warns once instead of on every frame, since they
		//	are reached from update loops that run at the frame rate.
		//
		// The state is leaked on purpose. The Application object is commonly a global, so the engine calls
		//	back into wi::physics during static destruction - unregistering the articulation step callback, for one -
		//	and a function local static container is liable to have been destroyed by then. Writing to it after its
		//	lifetime ends corrupts the heap: it showed up as this warning printing a second time for a string that
		//	was already in the set, followed by SIGABRT out of malloc at exit (2026-09-12, reproduced at exit=134).
		void unimplemented(const char* what)
		{
			static std::mutex* locker = new std::mutex();
			static wi::vector<std::string>* reported = new wi::vector<std::string>();
			std::scoped_lock lock(*locker);
			for (const std::string& seen : *reported)
			{
				if (seen == what)
					return;
			}
			reported->push_back(what);
			wilog_warning("wiPhysics_PhysX: %s is not implemented by the PhysX backend", what);
		}

		inline PxVec3 cast(const XMFLOAT3& v) { return PxVec3(v.x, v.y, v.z); }
		inline PxQuat cast(const XMFLOAT4& v) { return PxQuat(v.x, v.y, v.z, v.w); }
		inline XMFLOAT3 cast(const PxVec3& v) { return XMFLOAT3(v.x, v.y, v.z); }
		inline XMFLOAT4 cast(const PxQuat& v) { return XMFLOAT4(v.x, v.y, v.z, v.w); }

		// Continuous collision has to be asked for in three places in PhysX, and the Jolt backend gets it for free
		//	(EMotionQuality::LinearCast on every body). Without it a box dropped from a few metres tunnels straight
		//	through a height field at 60 Hz - measured 2026-09-12, the box that fell furthest ended up at -4417 m.
		//	The scene flag and the body flag are not enough on their own: the filter shader has to put
		//	eDETECT_CCD_CONTACT on the pair, which PxDefaultSimulationFilterShader does not do.
		PxFilterFlags SimulationFilterShader(
			PxFilterObjectAttributes attributes0, PxFilterData filterData0,
			PxFilterObjectAttributes attributes1, PxFilterData filterData1,
			PxPairFlags& pairFlags, const void* constantBlock, PxU32 constantBlockSize)
		{
			if (PxFilterObjectIsTrigger(attributes0) || PxFilterObjectIsTrigger(attributes1))
			{
				pairFlags = PxPairFlag::eTRIGGER_DEFAULT;
				return PxFilterFlag::eDEFAULT;
			}
			// The contact notifications are what feeds the per shape force sensors (foot forces). Asking for them
			//	on every pair costs a callback per touching pair per step; the callback returns immediately unless
			//	one of the two shapes carries a sensor, which is what the Jolt backend's listener does too.
			pairFlags = PxPairFlag::eCONTACT_DEFAULT | PxPairFlag::eDETECT_CCD_CONTACT
				| PxPairFlag::eNOTIFY_TOUCH_FOUND | PxPairFlag::eNOTIFY_TOUCH_PERSISTS | PxPairFlag::eNOTIFY_CONTACT_POINTS;
			return PxFilterFlag::eDEFAULT;
		}

		// The SDK singletons. Created once on first use and never released: PxPhysics::release() tears down every
		//	scene and actor still alive, and the scenes are owned by wi::scene::Scene objects whose destruction order
		//	against a function local static is not something this backend gets to choose.
		struct SDK
		{
			PxDefaultAllocator allocator;
			PxDefaultErrorCallback error_callback;
			PxFoundation* foundation = nullptr;
			PxPhysics* physics = nullptr;
			PxCookingParams cooking = PxCookingParams(PxTolerancesScale());
		};
		SDK& GetSDK()
		{
			static std::mutex* locker = new std::mutex();
			static SDK* sdk = nullptr;
			std::scoped_lock lock(*locker);
			if (sdk == nullptr)
			{
				sdk = new SDK();
				sdk->foundation = PxCreateFoundation(PX_PHYSICS_VERSION, sdk->allocator, sdk->error_callback);
				sdk->physics = PxCreatePhysics(PX_PHYSICS_VERSION, *sdk->foundation, PxTolerancesScale(), false);
				// The default 0.0007 merges faces that are nearly coplanar, so a hull asked for 16 sides can come
				//	back with 8 (measured on the wheel probe, 2026-09-11). Cylinders are approximated by hulls here,
				//	and the face count is exactly what governs their rolling behaviour, so keep every face.
				sdk->cooking.planeTolerance = 0;
			}
			return *sdk;
		}

		// Contact accumulator for a sensor shape. The engine reports per shape contact forces (foot force sensors on
		//	a robot), and PhysX hands them over as impulses in onContact, so they are summed per step and divided by
		//	the step length. Matches wiPhysics_Jolt's ContactAccumulator.
		struct ContactAccumulator
		{
			PxVec3 force = PxVec3(0.0f);
			uint32_t count = 0;
			wi::SpinLock lock;

			void Reset()
			{
				lock.lock();
				force = PxVec3(0.0f);
				count = 0;
				lock.unlock();
			}
			void Add(const PxVec3& f)
			{
				lock.lock();
				force += f;
				count++;
				lock.unlock();
			}
		};

		// Identifies the owner of a PhysX actor. Rigid bodies and articulation links are different objects on this
		//	side, so both embed one of these and hand its address to PxActor::userData.
		struct ActorUserData
		{
			Entity entity = INVALID_ENTITY;
		};

		struct SensorContactCallback : public PxSimulationEventCallback
		{
			float dt = 1.0f / 60.0f;

			void onContact(const PxContactPairHeader& header, const PxContactPair* pairs, PxU32 nbPairs) override
			{
				for (PxU32 i = 0; i < nbPairs; ++i)
				{
					const PxContactPair& pair = pairs[i];
					ContactAccumulator* a0 = pair.shapes[0] != nullptr ? (ContactAccumulator*)pair.shapes[0]->userData : nullptr;
					ContactAccumulator* a1 = pair.shapes[1] != nullptr ? (ContactAccumulator*)pair.shapes[1]->userData : nullptr;
					if (a0 == nullptr && a1 == nullptr)
						continue;
					if ((pair.events & (PxPairFlag::eNOTIFY_TOUCH_FOUND | PxPairFlag::eNOTIFY_TOUCH_PERSISTS)) == 0)
						continue;
					PxContactPairPoint points[64];
					const PxU32 n = pair.extractContacts(points, 64);
					PxVec3 impulse(0.0f);
					for (PxU32 p = 0; p < n; ++p)
					{
						impulse += points[p].impulse;
					}
					if (impulse.isZero())
						continue;
					// The impulse is reported as acting on the first shape's actor; the second one gets the opposite
					const PxVec3 force = impulse / std::max(1e-6f, dt);
					if (a0 != nullptr)
						a0->Add(force);
					if (a1 != nullptr)
						a1->Add(-force);
				}
			}
			void onConstraintBreak(PxConstraintInfo*, PxU32) override {}
			void onWake(PxActor**, PxU32) override {}
			void onSleep(PxActor**, PxU32) override {}
			void onTrigger(PxTriggerPair*, PxU32) override {}
			void onAdvance(const PxRigidBody* const*, const PxTransform*, const PxU32) override {}
		};

		// Shape cache, same idea as the Jolt backend: cooking a hull or a mesh is expensive and many rigid bodies
		//	share one MeshComponent. Not wired up yet - stage B creates one cooked mesh per body, which is correct
		//	but wasteful for scenes with many instances of the same mesh.
		struct PhysicsShapeCache
		{
			wi::unordered_map<const void*, PxBase*> cache;
			wi::SpinLock lock;
		};

		struct Articulation;

		struct PhysicsScene
		{
			PxScene* scene = nullptr;
			SensorContactCallback contact_callback;
			wi::vector<Articulation*> articulations;	// registered by AddArticulation, stepped by StepPhysicsScene
			PxDefaultCpuDispatcher* dispatcher = nullptr;
			PhysicsShapeCache physics_shape_cache;
			float accumulator = 0;
			float alpha = 0;
			bool activate_all_rigid_bodies = false;
			uint32_t steps_last_frame = 0;
			double sim_time = 0;
			uint64_t step_count = 0;
			wi::unordered_map<Entity, wi::physics::ArticulationStepCallback> articulation_callbacks;

			// Asynchronous simulation: a dedicated thread steps the scene at TIMESTEP intervals of wall clock time.
			//	The robot controller runs in the step callback, so the steps have to be paced by the clock rather
			//	than by the render frame - a frame's worth of steps taken back to back is a different experiment
			//	(measured 2026-09-12: LowState came out at 4064 Hz in bursts instead of 500 Hz).
			mutable std::recursive_mutex step_mutex;
			// PxScene is not safe for concurrent writes. The rigid body creation runs as parallel jobs inside the update, where
			//	StepLock is skipped (the update already holds step_mutex), so actor insertion / removal is serialized here instead
			std::mutex scene_write_mutex;
			std::atomic<bool> update_in_progress = false;
			std::atomic<bool> step_thread_running = false;
			std::atomic<bool> step_thread_quit = false;
			std::thread step_thread;
			std::atomic<uint64_t> steps_since_readback = 0;
			std::atomic<double> stat_wall_lag_ms = 0;
			std::atomic<double> stat_step_ms_acc = 0;
			std::atomic<uint32_t> stat_step_count = 0;
			std::atomic<double> stat_step_ms_max = 0;
			std::atomic<uint32_t> stat_resync_count = 0;

			void StartStepThread();
			void StopStepThread()
			{
				if (!step_thread_running)
					return;
				step_thread_quit = true;
				if (step_thread.joinable())
				{
					step_thread.join();
				}
				step_thread_running = false;
			}

			~PhysicsScene()
			{
				StopStepThread();
				if (scene != nullptr)
				{
					scene->release();
					scene = nullptr;
				}
				if (dispatcher != nullptr)
				{
					dispatcher->release();
					dispatcher = nullptr;
				}
			}
		};

		// Serializes with the stepping thread. A no-op while there is no such thread (stage D).
		struct StepLock
		{
			const PhysicsScene* physics_scene = nullptr;
			StepLock(const PhysicsScene& scene)
			{
				if (scene.step_thread_running && !scene.update_in_progress)
				{
					physics_scene = &scene;
					physics_scene->step_mutex.lock();
				}
			}
			~StepLock()
			{
				if (physics_scene != nullptr)
					physics_scene->step_mutex.unlock();
			}
		};
		struct UpdateScope
		{
			PhysicsScene& physics_scene;
			bool locked = false;
			UpdateScope(PhysicsScene& scene) : physics_scene(scene)
			{
				if (physics_scene.step_thread_running)
				{
					physics_scene.step_mutex.lock();
					locked = true;
				}
				physics_scene.update_in_progress = true;
			}
			~UpdateScope()
			{
				physics_scene.update_in_progress = false;
				if (locked)
					physics_scene.step_mutex.unlock();
			}
		};

		PhysicsScene& GetPhysicsScene(Scene& scene)
		{
			if (scene.physics_scene == nullptr)
			{
				SDK& sdk = GetSDK();
				auto physics_scene = wi::allocator::make_shared_single<PhysicsScene>();

				PxSceneDesc desc(sdk.physics->getTolerancesScale());
				desc.gravity = cast(scene.weather.gravity);
				physics_scene->dispatcher = PxDefaultCpuDispatcherCreate(std::max(1u, std::thread::hardware_concurrency() / 2));
				desc.cpuDispatcher = physics_scene->dispatcher;
				desc.filterShader = SimulationFilterShader;
				desc.flags |= PxSceneFlag::eENABLE_CCD;
				// Solver type. PGS (the SDK default) is the default here too: on the Go2 articulation TGS reports
				//	spurious joint velocities (dq inconsistent with the position update) at rest, on touchdown and
				//	under a 366:1 mass ratio impact, 10-20x worse than PGS in every case measured, and the controller
				//	reads dq. WICKED_PHYSX_SOLVER=tgs switches to TGS (=pgs is accepted too): the
				//	solver type is one of the things this backend exists to compare, and it is a scene creation
				//	parameter, so an environment variable keeps it out of the backend-neutral API while still being
				//	settable per run.
				desc.solverType = PxSolverType::ePGS;
				const char* solver_env = std::getenv("WICKED_PHYSX_SOLVER");
				if (solver_env != nullptr && std::strcmp(solver_env, "tgs") == 0)
				{
					desc.solverType = PxSolverType::eTGS;
				}
				wilog("wiPhysics_PhysX: scene solver %s", desc.solverType == PxSolverType::eTGS ? "TGS" : "PGS");
				physics_scene->scene = sdk.physics->createScene(desc);
				physics_scene->scene->setSimulationEventCallback(&physics_scene->contact_callback);

				scene.physics_scene = physics_scene;
			}
			return *(PhysicsScene*)scene.physics_scene.get();
		}

		// What CreateRigidBodyShape produces. PhysX has no standalone shape object the way Jolt does: geometry is a
		//	value and the shape only exists attached to an actor, so the geometry plus its local pose is what gets
		//	carried around. The cooked mesh behind a hull / mesh / height field is refcounted by the SDK.
		struct ShapeDesc
		{
			PxGeometryHolder geometry;
			PxTransform local_pose = PxTransform(PxIdentity);
			bool static_only = false;	// triangle meshes and height fields cannot back a dynamic actor
			bool valid = false;
		};

		struct RigidBody
		{
			wi::allocator::shared_ptr<void> physics_scene;
			Entity entity = INVALID_ENTITY;
			ShapeDesc shape;
			PxRigidActor* actor = nullptr;
			PxMaterial* material = nullptr;	// owned, so that friction and restitution are per body as the component says
			ActorUserData user_data;		// what PxActor::userData points at

			// property tracking:
			float friction = 0;
			float restitution = 0;
			bool is_dynamic = false;
			bool is_kinematic = false;
			bool start_deactivated = false;
			bool was_active_prev_frame = false;
			bool teleporting = false;
			PxVec3 initial_position = PxVec3(0.0f);
			PxQuat initial_rotation = PxQuat(PxIdentity);

			XMFLOAT4X4 parentMatrix = wi::math::IDENTITY_MATRIX;
			XMFLOAT4X4 parentMatrixInverse = wi::math::IDENTITY_MATRIX;

			PxVec3 prev_position = PxVec3(0.0f);
			PxQuat prev_rotation = PxQuat(PxIdentity);

			PxVec3 local_offset = PxVec3(0.0f);

			void Delete()
			{
				if (physics_scene != nullptr && actor != nullptr)
				{
					PhysicsScene* px_physics_scene = (PhysicsScene*)physics_scene.get();
					StepLock step_lock(*px_physics_scene);
					std::scoped_lock write_lock(px_physics_scene->scene_write_mutex);
					if (px_physics_scene->scene != nullptr)
					{
						px_physics_scene->scene->removeActor(*actor);
					}
					actor->release();
				}
				actor = nullptr;
				if (material != nullptr)
				{
					material->release();
					material = nullptr;
				}
			}
			~RigidBody()
			{
				Delete();
			}
		};

		RigidBody& GetRigidBody(RigidBodyPhysicsComponent& physicscomponent)
		{
			if (physicscomponent.physicsobject == nullptr)
			{
				physicscomponent.physicsobject = wi::allocator::make_shared<RigidBody>();
			}
			return *(RigidBody*)physicscomponent.physicsobject.get();
		}

		// PhysX capsules lie along X, the engine (and Jolt) put them along Y.
		const PxQuat kCapsuleYAxis = PxQuat(PxHalfPi, PxVec3(0, 0, 1));

		PxConvexMesh* CookHull(const PxVec3* points, uint32_t count)
		{
			PxConvexMeshDesc desc;
			desc.points.count = count;
			desc.points.stride = sizeof(PxVec3);
			desc.points.data = points;
			desc.flags = PxConvexFlag::eCOMPUTE_CONVEX;
			PxConvexMeshCookingResult::Enum result = PxConvexMeshCookingResult::eSUCCESS;
			PxConvexMesh* mesh = PxCreateConvexMesh(GetSDK().cooking, desc, *PxGetStandaloneInsertionCallback(), &result);
			if (mesh == nullptr)
			{
				wilog_error("wiPhysics_PhysX: convex cooking failed (result %d, %u points)", (int)result, count);
			}
			return mesh;
		}

		// A cylinder along Y as a prism. PhysX has no cylinder primitive; PxCustomGeometry has a CPU-only cylinder
		//	callback, which is the fallback if the facet count turns out to matter for foot contact (stage E).
		PxConvexMesh* CookCylinder(float half_height, float radius, uint32_t segments)
		{
			wi::vector<PxVec3> points;
			points.reserve(segments * 2);
			for (uint32_t i = 0; i < segments; ++i)
			{
				const float a = (float)i / (float)segments * XM_2PI;
				const float x = std::cos(a) * radius;
				const float z = std::sin(a) * radius;
				points.push_back(PxVec3(x, -half_height, z));
				points.push_back(PxVec3(x, half_height, z));
			}
			return CookHull(points.data(), (uint32_t)points.size());
		}

		// ---- Articulation ------------------------------------------------------------------------------------
		// A tree of links joined by revolute / fixed joints. This is the part the robot simulator exists for, and
		//	the reason for having a PhysX backend at all: PxArticulationReducedCoordinate solves the tree in joint
		//	coordinates, where Jolt has to express the same thing as bodies plus constraints.

		bool CollectArticulationLinks(const Scene& scene, Entity root_entity, wi::vector<Entity>& link_entities, wi::vector<int>& parent_indices)
		{
			link_entities.clear();
			parent_indices.clear();
			if (!scene.articulation_links.Contains(root_entity))
				return false;
			link_entities.push_back(root_entity);
			parent_indices.push_back(-1);

			// Breadth first, so the result is parent-first - which is also the order PhysX needs for createLink
			for (size_t i = 0; i < link_entities.size(); ++i)
			{
				const Entity parent_entity = link_entities[i];
				for (size_t j = 0; j < scene.articulation_links.GetCount(); ++j)
				{
					const ArticulationLinkComponent& link = scene.articulation_links[j];
					if (link.parent != parent_entity)
						continue;
					const Entity child_entity = scene.articulation_links.GetEntity(j);
					if (child_entity == root_entity)
						return false; // cycle
					for (Entity e : link_entities)
					{
						if (e == child_entity)
							return false; // cycle
					}
					link_entities.push_back(child_entity);
					parent_indices.push_back((int)i);
				}
			}
			return true;
		}

		// Rotation that takes the joint axis onto the twist axis (PhysX drives eTWIST, which is local X)
		PxQuat JointAxisFrame(const XMFLOAT3& axis_in)
		{
			PxVec3 x = cast(axis_in);
			if (x.normalizeSafe() <= 0)
				x = PxVec3(1, 0, 0);
			const PxVec3 up = PxAbs(x.z) < 0.9f ? PxVec3(0, 0, 1) : PxVec3(0, 1, 0);
			const PxVec3 y = up.cross(x).getNormalized();
			const PxVec3 z = x.cross(y);
			return PxQuat(PxMat33(x, y, z)).getNormalized();
		}

		struct Articulation
		{
			wi::allocator::shared_ptr<void> physics_scene;
			Entity root_entity = INVALID_ENTITY;
			PxArticulationReducedCoordinate* articulation = nullptr;
			PxArticulationCache* cache = nullptr;

			struct Link
			{
				Entity entity = INVALID_ENTITY;
				int parent = -1;
				ArticulationLinkComponent::JointType joint_type = ArticulationLinkComponent::JointType::Fixed;
				PxArticulationLink* link = nullptr;
				PxArticulationJointReducedCoordinate* joint = nullptr;
				PxMaterial* material = nullptr;
				ActorUserData user_data;
				uint32_t link_index = 0;		// PhysX internal link index (indexes the per link cache arrays)
				uint32_t dof_offset = 0;	// index into the cache's joint arrays
				bool has_dof = false;
				// The joint zero of a reduced coordinate joint is the pose the links were created in, which is the
				//	pose the TransformComponents carry, which is initial_position in engine terms. Same bookkeeping
				//	as the Jolt backend, so both report joint angles in the same convention.
				float angle_offset = 0;
				float mass = 0;
				float feedforward = 0;
				PxTransform creation_pose = PxTransform(PxIdentity);	// world pose of the link frame when it was created
				wi::vector<std::unique_ptr<ContactAccumulator>> sensors;	// indexed by Shape::sensor_id

				XMFLOAT4X4 parentMatrix = wi::math::IDENTITY_MATRIX;
				XMFLOAT4X4 parentMatrixInverse = wi::math::IDENTITY_MATRIX;

				// Measured every step:
				float position = 0;
				float velocity = 0;
				float motor_force = 0;
				float limit_force = 0;
				PxVec3 joint_force = PxVec3(0.0f);
				PxVec3 external_force = PxVec3(0.0f);
				PxVec3 prev_com_velocity = PxVec3(0.0f);
				bool has_prev_com_velocity = false;
				wi::vector<ArticulationLinkComponent::ContactSensor> contact_snapshot;
			};
			wi::vector<std::unique_ptr<Link>> links;	// parent-first, links[0] is the root

			bool external_drive = false;
			wi::vector<wi::physics::ArticulationLinkCommand> commands;
			wi::physics::ArticulationStepState step_state;

			PxVec3 root_linear_velocity = PxVec3(0.0f);
			PxVec3 root_angular_velocity = PxVec3(0.0f);
			PxVec3 root_linear_acceleration = PxVec3(0.0f);
			PxVec3 prev_root_velocity = PxVec3(0.0f);
			bool has_prev_root_velocity = false;
			wi::vector<PxVec3> external_forces_scratch;

			~Articulation()
			{
				if (physics_scene != nullptr)
				{
					PhysicsScene& scene = *(PhysicsScene*)physics_scene.get();
					StepLock step_lock(scene);
					for (size_t i = 0; i < scene.articulations.size(); ++i)
					{
						if (scene.articulations[i] == this)
						{
							scene.articulations[i] = scene.articulations.back();
							scene.articulations.pop_back();
							break;
						}
					}
					if (cache != nullptr)
					{
						cache->release();
						cache = nullptr;
					}
					if (articulation != nullptr && scene.scene != nullptr)
					{
						scene.scene->removeArticulation(*articulation);
					}
				}
				if (articulation != nullptr)
				{
					articulation->release();	// releases its links and their exclusive shapes
					articulation = nullptr;
				}
				for (auto& link : links)
				{
					if (link->material != nullptr)
					{
						link->material->release();
						link->material = nullptr;
					}
				}
				links.clear();
			}
		};

		Articulation& GetArticulation(ArticulationComponent& physicscomponent)
		{
			if (physicscomponent.physicsobject == nullptr)
			{
				physicscomponent.physicsobject = wi::allocator::make_shared<Articulation>();
			}
			return *(Articulation*)physicscomponent.physicsobject.get();
		}

		void ApplyLinkDrive(Articulation::Link& link, const wi::physics::ArticulationLinkCommand& cmd)
		{
			link.feedforward = cmd.feedforward_force;
			if (link.joint == nullptr || !link.has_dof)
				return;
			const float stiffness = std::max(0.0f, cmd.stiffness);
			const float damping = std::max(0.0f, cmd.damping);
			const float max_force = cmd.max_force >= FLT_MAX ? PX_MAX_F32 : std::max(0.0f, cmd.max_force);
			// PhysX solves the drive implicitly with the step, which is the same implicit PD that Jolt's motor
			//	spring does. maxForce is a force because of PxArticulationFlag::eDRIVE_LIMITS_ARE_FORCES.
			const PxArticulationDriveType::Enum type = (stiffness > 0 || damping > 0)
				? PxArticulationDriveType::eFORCE : PxArticulationDriveType::eNONE;
			link.joint->setDriveParams(PxArticulationAxis::eTWIST, PxArticulationDrive(stiffness, damping, max_force, type));
			link.joint->setDriveTarget(PxArticulationAxis::eTWIST, cmd.target_position - link.angle_offset);
			link.joint->setDriveVelocity(PxArticulationAxis::eTWIST, cmd.target_velocity);
		}

		void AddArticulation(Scene& scene, Entity root_entity, ArticulationComponent& physicscomponent)
		{
			wi::vector<Entity> link_entities;
			wi::vector<int> parent_indices;
			if (!CollectArticulationLinks(scene, root_entity, link_entities, parent_indices))
			{
				wilog_warning("AddArticulation: invalid link tree for root entity %u (root has no ArticulationLinkComponent, or there is a cycle)", (uint32_t)root_entity);
				return;
			}

			physicscomponent.physicsobject.reset(); // delete previous
			Articulation& articulation = GetArticulation(physicscomponent);
			PhysicsScene& physics_scene = GetPhysicsScene(scene);
			SDK& sdk = GetSDK();
			articulation.physics_scene = scene.physics_scene;
			articulation.root_entity = root_entity;
			physics_scene.articulations.push_back(&articulation);

			articulation.articulation = sdk.physics->createArticulationReducedCoordinate();
			PxArticulationFlags flags = PxArticulationFlag::eDRIVE_LIMITS_ARE_FORCES;
			if (physicscomponent.IsFixBase())
				flags |= PxArticulationFlag::eFIX_BASE;
			if (physicscomponent.IsSelfCollisionAllDisabled())
				flags |= PxArticulationFlag::eDISABLE_SELF_COLLISION;
			articulation.articulation->setArticulationFlags(flags);
			articulation.articulation->setSolverIterationCounts(
				(PxU32)std::max(1u, physicscomponent.position_iterations > 0 ? physicscomponent.position_iterations : (uint32_t)SOLVER_POSITION_ITERATIONS),
				(PxU32)(physicscomponent.velocity_iterations > 0 ? physicscomponent.velocity_iterations : (uint32_t)SOLVER_VELOCITY_ITERATIONS));
			// A free pendulum stops dead at its turning point otherwise: the kinetic energy is zero there and PhysX
			//	puts the whole articulation to sleep. The Jolt backend sets mAllowSleeping = false on articulation
			//	bodies for the same reason.
			articulation.articulation->setSleepThreshold(0.0f);

			const uint32_t link_count = (uint32_t)link_entities.size();
			for (uint32_t i = 0; i < link_count; ++i)
			{
				const Entity entity = link_entities[i];
				const ArticulationLinkComponent* linkcomponent = scene.articulation_links.GetComponent(entity);
				const TransformComponent* transform = scene.transforms.GetComponent(entity);
				if (linkcomponent == nullptr || transform == nullptr)
				{
					wilog_warning("AddArticulation: link entity %u has no TransformComponent", (uint32_t)entity);
					break;
				}

				articulation.links.push_back(std::make_unique<Articulation::Link>());
				Articulation::Link& link = *articulation.links.back();
				link.entity = entity;
				link.parent = parent_indices[i];
				link.joint_type = linkcomponent->joint_type;
				link.user_data.entity = entity;

				// World pose of the link frame
				scene.locker.lock();
				const XMMATRIX parentMatrix = scene.ComputeParentMatrixRecursive(entity);
				scene.locker.unlock();
				XMStoreFloat4x4(&link.parentMatrix, parentMatrix);
				XMStoreFloat4x4(&link.parentMatrixInverse, XMMatrixInverse(nullptr, parentMatrix));
				XMFLOAT4X4 worldMatrix;
				XMStoreFloat4x4(&worldMatrix, transform->GetLocalMatrix() * parentMatrix);
				XMVECTOR S, R, T;
				XMMatrixDecompose(&S, &R, &T, XMLoadFloat4x4(&worldMatrix));
				XMFLOAT3 world_position;
				XMFLOAT4 world_rotation;
				XMStoreFloat3(&world_position, T);
				XMStoreFloat4(&world_rotation, R);
				const PxTransform link_pose(cast(world_position), cast(world_rotation).getNormalized());

				PxArticulationLink* parent_link = link.parent >= 0 ? articulation.links[link.parent]->link : nullptr;
				link.link = articulation.articulation->createLink(parent_link, link_pose);
				if (link.link == nullptr)
				{
					wilog_warning("AddArticulation: link creation failed on entity %u", (uint32_t)entity);
					articulation.links.pop_back();
					break;
				}
				link.link->userData = &link.user_data;
				link.creation_pose = link_pose;

				// Shapes. One PxShape per collision shape, placed by its pose in the link frame; PhysX aggregates
				//	them on the actor, so there is no compound object the way Jolt needs one.
				link.material = sdk.physics->createMaterial(linkcomponent->friction, linkcomponent->friction, linkcomponent->restitution);
				int32_t max_sensor_id = -1;
				for (const ArticulationLinkComponent::Shape& shape : linkcomponent->shapes)
				{
					max_sensor_id = std::max(max_sensor_id, shape.sensor_id);
				}
				for (int32_t sensor = 0; sensor <= max_sensor_id; ++sensor)
				{
					link.sensors.push_back(std::make_unique<ContactAccumulator>());
				}
				for (const ArticulationLinkComponent::Shape& shape : linkcomponent->shapes)
				{
					PxTransform pose(cast(shape.position), cast(shape.rotation).getNormalized());
					PxShape* px_shape = nullptr;
					switch (shape.type)
					{
					default:
					case RigidBodyPhysicsComponent::CollisionShape::BOX:
						px_shape = PxRigidActorExt::createExclusiveShape(*link.link,
							PxBoxGeometry(std::max(0.0001f, shape.halfextents.x), std::max(0.0001f, shape.halfextents.y), std::max(0.0001f, shape.halfextents.z)), *link.material);
						break;
					case RigidBodyPhysicsComponent::CollisionShape::SPHERE:
						px_shape = PxRigidActorExt::createExclusiveShape(*link.link,
							PxSphereGeometry(std::max(0.0001f, shape.radius)), *link.material);
						break;
					case RigidBodyPhysicsComponent::CollisionShape::CAPSULE:
						// PhysX capsules run along local X, the engine's along local Y
						pose = pose * PxTransform(PxVec3(0.0f), kCapsuleYAxis);
						px_shape = PxRigidActorExt::createExclusiveShape(*link.link,
							PxCapsuleGeometry(std::max(0.0001f, shape.radius), std::max(0.0001f, shape.height * 0.5f)), *link.material);
						break;
					case RigidBodyPhysicsComponent::CollisionShape::CYLINDER:
					{
						PxConvexMesh* hull = CookCylinder(std::max(0.0001f, shape.height * 0.5f), std::max(0.0001f, shape.radius), 16);
						if (hull != nullptr)
						{
							px_shape = PxRigidActorExt::createExclusiveShape(*link.link, PxConvexMeshGeometry(hull), *link.material);
						}
						break;
					}
					}
					if (px_shape == nullptr)
						continue;
					px_shape->setLocalPose(pose);
					if (shape.sensor_id >= 0 && shape.sensor_id < (int32_t)link.sensors.size())
					{
						px_shape->userData = link.sensors[shape.sensor_id].get();
					}
				}

				// Mass properties. PhysX wants the inertia diagonalized, with the rotation folded into the mass frame.
				float mass = linkcomponent->mass;
				if (mass <= 0)
				{
					wilog_warning("AddArticulation: link entity %u has non-positive mass, clamped to 0.001 kg", (uint32_t)entity);
					mass = 0.001f;
				}
				const XMFLOAT3X3& I = linkcomponent->inertia;
				const PxMat33 inertia(
					PxVec3(I._11, I._21, I._31),
					PxVec3(I._12, I._22, I._32),
					PxVec3(I._13, I._23, I._33));
				PxQuat mass_frame(PxIdentity);
				const PxVec3 diagonal = PxMassProperties::getMassSpaceInertia(inertia, mass_frame);
				link.link->setMass(mass);
				link.link->setCMassLocalPose(PxTransform(cast(linkcomponent->center_of_mass), mass_frame));
				link.link->setMassSpaceInertiaTensor(diagonal);
				link.link->setLinearDamping(0);
				link.link->setAngularDamping(0);	// PhysX defaults this to 0.05, which would quietly damp every link
				link.mass = (i == 0 && physicscomponent.IsFixBase()) ? 0 : mass;

				if (link.parent < 0)
					continue;

				// Joint
				PxArticulationJointReducedCoordinate* joint = link.link->getInboundJoint();
				link.joint = joint;
				const PxQuat axis_frame = JointAxisFrame(linkcomponent->axis);
				const PxTransform parent_pose(cast(linkcomponent->joint_position_parent),
					cast(linkcomponent->joint_rotation_parent).getNormalized() * axis_frame);
				joint->setParentPose(parent_pose);

				// A reduced coordinate joint reads zero where the two joint frames coincide, and the links were
				//	created at the pose the TransformComponents carry, which is initial_position - not zero. Giving
				//	PhysX the nominal child frame instead would make it enforce the joint at creation and rearrange
				//	the whole tree on the first step (measured 2026-09-12: the trunk rose from 0.29 m to 0.45 m and
				//	the left and right legs swapped sides). So derive the child frame's orientation from the pose
				//	the links are actually in, which is what wiPhysics_Jolt does for the hinge's child axes. The
				//	joint point itself is on the axis and therefore unaffected by the angle.
				const PxTransform& parent_world = articulation.links[link.parent]->creation_pose;
				const PxQuat joint_frame_world = parent_world.q * parent_pose.q;
				joint->setChildPose(PxTransform(cast(linkcomponent->joint_position_child),
					(link.creation_pose.q.getConjugate() * joint_frame_world).getNormalized()));
				link.angle_offset = linkcomponent->initial_position;

				// The joint point has to be the same place in world from both sides, or the links do not agree
				{
					const PxVec3 from_parent = parent_world.transform(parent_pose.p);
					const PxVec3 from_child = link.creation_pose.transform(cast(linkcomponent->joint_position_child));
					const float mismatch = (from_parent - from_child).magnitude();
					if (mismatch > 0.001f)
					{
						wilog_warning("AddArticulation: joint position mismatch of %f m between parent and child frames on link entity %u (link transforms are not consistent with the joint frames)", mismatch, (uint32_t)link.entity);
					}
				}

				switch (linkcomponent->joint_type)
				{
				case ArticulationLinkComponent::JointType::Revolute:
				{
					// Unwrapped: the joint position is not folded into (-pi, pi], so the limits can be given in the
					//	engine's own units without the clamping the Jolt hinge needs
					joint->setJointType(PxArticulationJointType::eREVOLUTE_UNWRAPPED);
					const bool has_limits = linkcomponent->limit_min > -XM_PI || linkcomponent->limit_max < XM_PI;
					if (has_limits)
					{
						joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
						joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(
							linkcomponent->limit_min - link.angle_offset, linkcomponent->limit_max - link.angle_offset));
					}
					else
					{
						joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
					}
					// Native, unlike the Jolt backend where the caller has to fold it into the link inertias (see GetBackendName)
					joint->setArmature(PxArticulationAxis::eTWIST, std::max(0.0f, linkcomponent->armature));
					// Two friction models coexist in 5.9 and the legacy one defaults to 0.05, not 0: left alone it
					//	damps every joint in proportion to the load it carries and a free pendulum stops at its first
					//	turning point. Setting the new absolute-effort parameters does not switch the old one off.
					joint->setFrictionCoefficient(0.0f);
					joint->setFrictionParams(PxArticulationAxis::eTWIST,
						PxJointFrictionParams(std::max(0.0f, linkcomponent->joint_friction), std::max(0.0f, linkcomponent->joint_friction), 0.0f));
					link.has_dof = true;
				}
				break;
				case ArticulationLinkComponent::JointType::Prismatic:
					wilog_warning("AddArticulation: prismatic joint is not implemented in the PhysX backend, link entity %u is fixed to its parent", (uint32_t)link.entity);
					[[fallthrough]];
				default:
				case ArticulationLinkComponent::JointType::Fixed:
					joint->setJointType(PxArticulationJointType::eFIX);
					break;
				}
			}

			if (articulation.links.size() != link_count)
			{
				wilog_warning("AddArticulation: some links could not be created, articulation of root entity %u is not created", (uint32_t)root_entity);
				physicscomponent.physicsobject.reset();
				return;
			}

			{
				StepLock step_lock(physics_scene);
				physics_scene.scene->addArticulation(*articulation.articulation);
			}
			// The cache can only be made once the articulation is in a scene
			articulation.cache = articulation.articulation->createCache();

			// Cache index bookkeeping: the joint arrays are laid out in link index order, so a link's offset is the
			//	sum of the dofs of every link with a lower internal index
			for (auto& link_ptr : articulation.links)
			{
				Articulation::Link& link = *link_ptr;
				link.link_index = link.link->getLinkIndex();
			}
			for (auto& link_ptr : articulation.links)
			{
				Articulation::Link& link = *link_ptr;
				PxU32 offset = 0;
				for (auto& other_ptr : articulation.links)
				{
					if (other_ptr->link_index < link.link_index)
						offset += other_ptr->link->getInboundJointDof();
				}
				link.dof_offset = offset;
				link.has_dof = link.link->getInboundJointDof() > 0;
			}

			physicscomponent.link_count = link_count;
			physicscomponent.SetRefreshParametersNeeded(true);
		}

		// Before a step: the drive commands from the step callback, the feedforward joint efforts, and a clean slate
		//	for the contact sensors
		void ArticulationPreStep(Articulation& articulation)
		{
			if (articulation.articulation == nullptr || articulation.cache == nullptr)
				return;
			if (articulation.external_drive)
			{
				for (size_t l = 0; l < articulation.links.size() && l < articulation.commands.size(); ++l)
				{
					ApplyLinkDrive(*articulation.links[l], articulation.commands[l]);
				}
			}
			for (auto& link_ptr : articulation.links)
			{
				for (auto& sensor : link_ptr->sensors)
				{
					sensor->Reset();
				}
			}
			// Feedforward joint effort. Set every step, since PhysX clears applied joint forces after each one.
			const PxU32 dofs = articulation.articulation->getDofs();
			bool any = false;
			for (PxU32 d = 0; d < dofs; ++d)
			{
				articulation.cache->jointForce[d] = 0;
			}
			for (auto& link_ptr : articulation.links)
			{
				const Articulation::Link& link = *link_ptr;
				if (!link.has_dof || link.dof_offset >= dofs)
					continue;
				articulation.cache->jointForce[link.dof_offset] = link.feedforward;
				any = any || link.feedforward != 0;
			}
			if (any)
			{
				articulation.articulation->applyCache(*articulation.cache, PxArticulationCacheFlag::eFORCE);
			}
		}

		// After a step: joint state, joint forces, contact snapshots, external force estimates, root state, callback
		void ArticulationPostStep(PhysicsScene& physics_scene, Articulation& articulation, float dt)
		{
			if (articulation.links.empty() || articulation.articulation == nullptr || articulation.cache == nullptr)
				return;

			// Joint position and velocity come out of the same cache read on purpose. Taking the angle from the link
			//	poses and the velocity from the link velocities makes the two disagree exactly where the solver
			//	corrects position, which is what the Jolt backend does and what `simprobe consist` measures.
			articulation.articulation->copyInternalStateToCache(*articulation.cache,
				PxArticulationCacheFlag::ePOSITION | PxArticulationCacheFlag::eVELOCITY | PxArticulationCacheFlag::eLINK_INCOMING_JOINT_FORCE);

			const PxVec3 gravity = physics_scene.scene->getGravity();
			const PxU32 dofs = articulation.articulation->getDofs();

			for (auto& link_ptr : articulation.links)
			{
				Articulation::Link& link = *link_ptr;
				if (link.link == nullptr)
					continue;

				link.contact_snapshot.resize(link.sensors.size());
				for (size_t s = 0; s < link.sensors.size(); ++s)
				{
					ContactAccumulator& sensor = *link.sensors[s];
					sensor.lock.lock();
					link.contact_snapshot[s].force = cast(sensor.force);
					link.contact_snapshot[s].contact_count = sensor.count;
					sensor.lock.unlock();
				}

				link.joint_force = PxVec3(0.0f);
				link.motor_force = 0;
				link.limit_force = 0;
				if (link.parent >= 0)
				{
					if (link.has_dof && link.dof_offset < dofs)
					{
						link.position = articulation.cache->jointPosition[link.dof_offset] + link.angle_offset;
						link.velocity = articulation.cache->jointVelocity[link.dof_offset];
					}
					// The incoming joint force is the total transmitted from the parent, given in the child joint
					//	frame, so its twist component is the joint torque (drive, limit and friction together).
					//	Jolt can separate motor from limit; PhysX reports the sum, which is what tauEst wants anyway.
					const PxSpatialForce& spatial = articulation.cache->linkIncomingJointForce[link.link_index];
					link.motor_force = spatial.torque.x;
					const PxTransform joint_frame = link.link->getGlobalPose() * link.joint->getChildPose();
					link.joint_force = joint_frame.q.rotate(spatial.force);
				}
			}

			// External force per link: m * (a - g) - F_joint_in + sum(F_joint_out of the children). Same estimator
			//	as the Jolt backend, so the two report the same quantity.
			{
				wi::vector<PxVec3>& external_forces = articulation.external_forces_scratch;
				external_forces.assign(articulation.links.size(), PxVec3(0.0f));
				for (size_t l = 0; l < articulation.links.size(); ++l)
				{
					Articulation::Link& link = *articulation.links[l];
					if (link.link == nullptr)
						continue;
					const PxVec3 com_velocity = link.link->getLinearVelocity();
					if (link.has_prev_com_velocity && link.mass > 0)
					{
						const PxVec3 acceleration = (com_velocity - link.prev_com_velocity) / dt;
						external_forces[l] += (acceleration - gravity) * link.mass;
					}
					link.prev_com_velocity = com_velocity;
					link.has_prev_com_velocity = true;
					if (link.parent >= 0)
					{
						external_forces[l] -= link.joint_force;
						external_forces[link.parent] += link.joint_force;
					}
				}
				for (size_t l = 0; l < articulation.links.size(); ++l)
				{
					articulation.links[l]->external_force = external_forces[l];
				}
			}

			// Root state, at the link frame origin rather than the centre of mass
			const Articulation::Link& root = *articulation.links[0];
			PxTransform root_pose(PxIdentity);
			if (root.link != nullptr)
			{
				root_pose = root.link->getGlobalPose();
				const PxVec3 com_world = root.link->getGlobalPose().transform(root.link->getCMassLocalPose().p);
				const PxVec3 angular = root.link->getAngularVelocity();
				const PxVec3 linear = root.link->getLinearVelocity() + angular.cross(root_pose.p - com_world);
				articulation.root_linear_velocity = linear;
				articulation.root_angular_velocity = angular;
				if (articulation.has_prev_root_velocity)
				{
					articulation.root_linear_acceleration = (linear - articulation.prev_root_velocity) / dt - gravity;
				}
				articulation.prev_root_velocity = linear;
				articulation.has_prev_root_velocity = true;
			}

			auto it = physics_scene.articulation_callbacks.find(articulation.root_entity);
			if (it == physics_scene.articulation_callbacks.end() || !it->second)
			{
				articulation.external_drive = false;
				return;
			}
			wi::physics::ArticulationStepState& state = articulation.step_state;
			state.time = physics_scene.sim_time;
			state.step = physics_scene.step_count;
			state.dt = dt;
			state.root_position = cast(root_pose.p);
			state.root_rotation = cast(root_pose.q.getNormalized());
			state.root_linear_velocity = cast(articulation.root_linear_velocity);
			state.root_angular_velocity = cast(articulation.root_angular_velocity);
			state.root_linear_acceleration = cast(articulation.root_linear_acceleration);
			state.links.resize(articulation.links.size());
			for (size_t l = 0; l < articulation.links.size(); ++l)
			{
				const Articulation::Link& link = *articulation.links[l];
				wi::physics::ArticulationStepState::Link& out = state.links[l];
				out.entity = link.entity;
				out.position = link.position;
				out.velocity = link.velocity;
				out.motor_force = link.motor_force;
				out.limit_force = link.limit_force;
				out.joint_force = cast(link.joint_force);
				out.external_force = cast(link.external_force);
				out.contacts.resize(link.contact_snapshot.size());
				for (size_t s = 0; s < link.contact_snapshot.size(); ++s)
				{
					out.contacts[s].force = link.contact_snapshot[s].force;
					out.contacts[s].count = link.contact_snapshot[s].contact_count;
				}
			}
			articulation.commands.resize(articulation.links.size());
			it->second(state, articulation.commands);
			articulation.external_drive = true;
		}

	}

	const char* GetBackendName()
	{
		return "PhysX";
	}

	void Initialize()
	{
		wi::Timer timer;
		GetSDK();
		wilog("wi::physics Initialized [PhysX %d.%d.%d] (%d ms)", PX_PHYSICS_VERSION_MAJOR, PX_PHYSICS_VERSION_MINOR,
			PX_PHYSICS_VERSION_BUGFIX, (int)std::round(timer.elapsed()));
	}

	void CreateRigidBodyShape(RigidBodyPhysicsComponent& physicscomponent, const XMFLOAT3& scale_local, const MeshComponent* mesh)
	{
		RigidBody& physicsobject = GetRigidBody(physicscomponent);
		ShapeDesc& out = physicsobject.shape;
		out = ShapeDesc();

		switch (physicscomponent.shape)
		{
		case RigidBodyPhysicsComponent::CollisionShape::BOX:
		{
			out.geometry = PxBoxGeometry(
				std::max(0.0001f, physicscomponent.box.halfextents.x * scale_local.x),
				std::max(0.0001f, physicscomponent.box.halfextents.y * scale_local.y),
				std::max(0.0001f, physicscomponent.box.halfextents.z * scale_local.z));
			out.valid = true;
		}
		break;
		case RigidBodyPhysicsComponent::CollisionShape::SPHERE:
		{
			out.geometry = PxSphereGeometry(std::max(0.0001f, physicscomponent.sphere.radius * scale_local.x));
			out.valid = true;
		}
		break;
		case RigidBodyPhysicsComponent::CollisionShape::CAPSULE:
		{
			// The engine's capsule.height is the half height of the cylindrical part, same as Jolt's setting
			out.geometry = PxCapsuleGeometry(
				std::max(0.0001f, physicscomponent.capsule.radius * scale_local.x),
				std::max(0.0001f, physicscomponent.capsule.height * scale_local.y));
			out.local_pose = PxTransform(PxVec3(0.0f), kCapsuleYAxis);
			out.valid = true;
		}
		break;
		case RigidBodyPhysicsComponent::CollisionShape::CYLINDER:
		{
			PxConvexMesh* hull = CookCylinder(
				std::max(0.0001f, physicscomponent.capsule.height * scale_local.y),
				std::max(0.0001f, physicscomponent.capsule.radius * scale_local.x), 16);
			if (hull == nullptr)
				return;
			out.geometry = PxConvexMeshGeometry(hull);
			out.valid = true;
		}
		break;
		case RigidBodyPhysicsComponent::CollisionShape::CONVEX_HULL:
		{
			if (mesh == nullptr)
			{
				wilog_error("CreateRigidBodyShape failed: convex hull physics requested, but no MeshComponent provided!");
				return;
			}
			wi::vector<PxVec3> points;
			points.reserve(mesh->vertex_positions.size());
			for (const XMFLOAT3& pos : mesh->vertex_positions)
			{
				points.push_back(PxVec3(pos.x * scale_local.x, pos.y * scale_local.y, pos.z * scale_local.z));
			}
			PxConvexMesh* hull = CookHull(points.data(), (uint32_t)points.size());
			if (hull == nullptr)
				return;
			out.geometry = PxConvexMeshGeometry(hull);
			out.valid = true;
		}
		break;
		case RigidBodyPhysicsComponent::CollisionShape::TRIANGLE_MESH:
		{
			if (mesh == nullptr)
			{
				wilog_error("CreateRigidBodyShape failed: triangle mesh physics requested, but no MeshComponent provided!");
				return;
			}
			wi::vector<PxVec3> points;
			points.reserve(mesh->vertex_positions.size());
			for (const XMFLOAT3& pos : mesh->vertex_positions)
			{
				points.push_back(PxVec3(pos.x * scale_local.x, pos.y * scale_local.y, pos.z * scale_local.z));
			}
			wi::vector<uint32_t> indices;
			uint32_t first_subset = 0;
			uint32_t last_subset = 0;
			mesh->GetLODSubsetRange(physicscomponent.mesh_lod, first_subset, last_subset);
			for (uint32_t subsetIndex = first_subset; subsetIndex < last_subset; ++subsetIndex)
			{
				const MeshComponent::MeshSubset& subset = mesh->subsets[subsetIndex];
				const uint32_t* src = mesh->indices.data() + subset.indexOffset;
				for (uint32_t i = 0; i + 2 < subset.indexCount; i += 3)
				{
					// Same winding flip the Jolt backend applies, so both engines see the same face normals
					indices.push_back(src[i + 0]);
					indices.push_back(src[i + 2]);
					indices.push_back(src[i + 1]);
				}
			}
			PxTriangleMeshDesc desc;
			desc.points.count = (PxU32)points.size();
			desc.points.stride = sizeof(PxVec3);
			desc.points.data = points.data();
			desc.triangles.count = (PxU32)(indices.size() / 3);
			desc.triangles.stride = 3 * sizeof(uint32_t);
			desc.triangles.data = indices.data();
			PxTriangleMeshCookingResult::Enum result = PxTriangleMeshCookingResult::eSUCCESS;
			PxTriangleMesh* trimesh = PxCreateTriangleMesh(GetSDK().cooking, desc, *PxGetStandaloneInsertionCallback(), &result);
			if (trimesh == nullptr)
			{
				wilog_error("wiPhysics_PhysX: triangle mesh cooking failed (result %d)", (int)result);
				return;
			}
			out.geometry = PxTriangleMeshGeometry(trimesh);
			out.static_only = true;	// PhysX refuses a triangle mesh on a dynamic actor unless it carries an SDF
			out.valid = true;
		}
		break;
		case RigidBodyPhysicsComponent::CollisionShape::HEIGHTFIELD:
		{
			if (mesh == nullptr)
			{
				wilog_error("CreateRigidBodyShape failed: height field physics requested, but no MeshComponent provided!");
				return;
			}
			const size_t count = mesh->vertex_positions.size();
			const uint32_t dim = (uint32_t)std::lround(std::sqrt((double)count));
			if (dim < 2 || (size_t)dim * dim != count)
			{
				wilog_error("wiPhysics_PhysX: height field needs a square vertex grid, got %u vertices", (uint32_t)count);
				return;
			}

			// The engine hands over a grid of vertices, not a sample array, and the two engines index it the other
			//	way round: Jolt reads sample (x, z) at [z * dim + x], PhysX reads [row * dim + column] with the row
			//	running along X. Rather than assume, look at what the grid actually does between the first two
			//	vertices, so a change on the producing side (the height field mesh generator) cannot silently transpose terrain.
			const bool x_varies_first = std::abs(mesh->vertex_positions[1].x - mesh->vertex_positions[0].x) >
				std::abs(mesh->vertex_positions[1].z - mesh->vertex_positions[0].z);

			wi::primitive::AABB aabb;
			for (XMFLOAT3 pos : mesh->vertex_positions)
			{
				pos.x *= scale_local.x;
				pos.y *= scale_local.y;
				pos.z *= scale_local.z;
				aabb.AddPoint(pos);
			}
			const XMFLOAT3 aabb_min = aabb.getMin();
			const XMFLOAT3 half = aabb.getHalfWidth();
			const float height_min = aabb_min.y;
			const float height_range = std::max(1e-4f, half.y * 2);
			// 16 bit samples. Over a 100 m field with a few metres of relief this is well under a tenth of a
			//	millimetre, but it is a difference from Jolt's float samples and it is recorded rather than assumed.
			const float height_scale = height_range / 32767.0f;

			wi::vector<PxHeightFieldSample> samples;
			samples.resize(count);
			for (uint32_t row = 0; row < dim; ++row)
			{
				for (uint32_t col = 0; col < dim; ++col)
				{
					// row runs along X in PhysX
					const uint32_t src = x_varies_first ? (col * dim + row) : (row * dim + col);
					const float h = mesh->vertex_positions[src].y * scale_local.y - height_min;
					PxHeightFieldSample& s = samples[row * dim + col];
					s.height = (PxI16)std::lround(std::min(32767.0f, std::max(0.0f, h / height_scale)));
					s.materialIndex0 = 0;
					s.materialIndex1 = 0;
				}
			}

			PxHeightFieldDesc desc;
			desc.format = PxHeightFieldFormat::eS16_TM;
			desc.nbRows = dim;
			desc.nbColumns = dim;
			desc.samples.data = samples.data();
			desc.samples.stride = sizeof(PxHeightFieldSample);
			PxHeightField* hf = PxCreateHeightField(desc, *PxGetStandaloneInsertionCallback());
			if (hf == nullptr)
			{
				wilog_error("wiPhysics_PhysX: height field creation failed (%u x %u)", dim, dim);
				return;
			}
			const float row_scale = half.x * 2 / (float)(dim - 1);
			const float col_scale = half.z * 2 / (float)(dim - 1);
			out.geometry = PxHeightFieldGeometry(hf, PxMeshGeometryFlags(), height_scale, row_scale, col_scale);
			out.local_pose = PxTransform(PxVec3(aabb_min.x, height_min, aabb_min.z));
			out.static_only = true;
			out.valid = true;
			wilog("wiPhysics_PhysX: height field %u x %u, %.4f m cell, %.3f m relief quantized to %.3g m per step%s",
				dim, dim, row_scale, height_range, height_scale, x_varies_first ? "" : " (grid transposed)");
		}
		break;
		default:
			unimplemented("this collision shape");
			return;
		}
	}

	namespace
	{
		void AddRigidBody(Scene& scene, Entity entity, RigidBodyPhysicsComponent& physicscomponent,
			const TransformComponent& _transform, const MeshComponent* mesh)
		{
			RigidBody& physicsobject = GetRigidBody(physicscomponent);
			const bool refresh = physicsobject.actor != nullptr;
			physicsobject.Delete();

			TransformComponent transform = _transform;
			scene.locker.lock();
			const XMMATRIX parentMatrix = scene.ComputeParentMatrixRecursive(entity);
			scene.locker.unlock();
			transform.ApplyTransform();

			if (!physicsobject.shape.valid)
			{
				CreateRigidBodyShape(physicscomponent, transform.scale_local, mesh);
			}
			if (!physicsobject.shape.valid)
				return;

			if (physicscomponent.IsCharacterPhysics() || physicscomponent.IsVehicle())
			{
				unimplemented("character controllers and vehicles");
				return;
			}

			SDK& sdk = GetSDK();
			PhysicsScene& physics_scene = GetPhysicsScene(scene);

			physicsobject.physics_scene = scene.physics_scene;
			physicsobject.entity = entity;
			XMStoreFloat4x4(&physicsobject.parentMatrix, parentMatrix);
			XMStoreFloat4x4(&physicsobject.parentMatrixInverse, XMMatrixInverse(nullptr, parentMatrix));

			physicsobject.local_offset = cast(physicscomponent.local_offset);
			physicsobject.prev_position = cast(transform.GetPosition());
			physicsobject.prev_rotation = cast(transform.GetRotation()).getNormalized();
			if (!refresh)
			{
				physicsobject.initial_position = physicsobject.prev_position;
				physicsobject.initial_rotation = physicsobject.prev_rotation;
			}

			bool dynamic = physicscomponent.mass > 0;
			if (dynamic && physicsobject.shape.static_only)
			{
				unimplemented("dynamic bodies with a triangle mesh or height field shape (kept static)");
				dynamic = false;
			}
			const bool kinematic = dynamic && physicscomponent.IsKinematic();

			const PxTransform pose(physicsobject.local_offset + physicsobject.prev_position, physicsobject.prev_rotation);
			if (dynamic)
			{
				physicsobject.actor = sdk.physics->createRigidDynamic(pose);
			}
			else
			{
				physicsobject.actor = sdk.physics->createRigidStatic(pose);
			}
			if (physicsobject.actor == nullptr)
			{
				wilog_error("AddRigidBody failed: actor could not be created");
				return;
			}

			// Per body material: the component carries friction and restitution per rigid body, and a PxMaterial is
			//	the only place PhysX keeps them. Friction combines as the average of the two materials by default,
			//	which is neither Jolt's geometric mean nor MuJoCo's max - stage E sets this deliberately.
			physicsobject.material = sdk.physics->createMaterial(physicscomponent.friction, physicscomponent.friction, physicscomponent.restitution);
			physicsobject.friction = physicscomponent.friction;
			physicsobject.restitution = physicscomponent.restitution;

			PxShape* shape = PxRigidActorExt::createExclusiveShape(*physicsobject.actor, physicsobject.shape.geometry.any(), *physicsobject.material);
			if (shape == nullptr)
			{
				wilog_error("AddRigidBody failed: shape could not be attached");
				physicsobject.actor->release();
				physicsobject.actor = nullptr;
				return;
			}
			shape->setLocalPose(physicsobject.shape.local_pose);

			physicsobject.is_dynamic = dynamic;
			physicsobject.is_kinematic = kinematic;
			if (dynamic)
			{
				PxRigidDynamic* body = static_cast<PxRigidDynamic*>(physicsobject.actor);
				PxRigidBodyExt::setMassAndUpdateInertia(*body, physicscomponent.mass);
				body->setLinearDamping(physicscomponent.damping_linear);
				body->setAngularDamping(physicscomponent.damping_angular);
				body->setSolverIterationCounts((PxU32)std::max(1, SOLVER_POSITION_ITERATIONS), (PxU32)std::max(0, SOLVER_VELOCITY_ITERATIONS));
				if (kinematic)
				{
					body->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
				}
				// Matches Jolt's EMotionQuality::LinearCast, which that backend uses for every body
				body->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_CCD, true);
				if (physicscomponent.IsDisableDeactivation())
				{
					body->setSleepThreshold(0.0f);
				}
				if (physicscomponent.IsLocked2D())
				{
					unimplemented("2D locked rigid bodies");
				}
			}
			physicsobject.start_deactivated = physicscomponent.IsStartDeactivated();

			physicsobject.user_data.entity = entity;
			physicsobject.actor->userData = &physicsobject.user_data;
			{
				StepLock step_lock(physics_scene);
				std::scoped_lock write_lock(physics_scene.scene_write_mutex);
				physics_scene.scene->addActor(*physicsobject.actor);
				if (dynamic && !kinematic && physicsobject.start_deactivated)
				{
					static_cast<PxRigidDynamic*>(physicsobject.actor)->putToSleep();
				}
			}
			physicsobject.was_active_prev_frame = !physicsobject.start_deactivated;
		}

		void StepPhysicsScene(PhysicsScene& physics_scene);

		// Steps the scene at TIMESTEP intervals of wall clock time. Same structure as the Jolt backend's, so that
		//	the pacing, the lag statistics and the resync rule are the same experiment on both.
		void StepThreadMain(PhysicsScene* physics_scene)
		{
			using clock = std::chrono::steady_clock;
			bool synced = false;
			clock::time_point next = clock::now();
			while (!physics_scene->step_thread_quit)
			{
				if (!IsSimulationEnabled())
				{
					synced = false; // the schedule is rebuilt when resuming
					std::this_thread::sleep_for(std::chrono::milliseconds(1));
					continue;
				}
				const auto period = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(TIMESTEP));
				if (!synced)
				{
					next = clock::now();
					synced = true;
				}

				const clock::time_point step_begin = clock::now();
				{
					std::lock_guard<std::recursive_mutex> lock(physics_scene->step_mutex);
					StepPhysicsScene(*physics_scene);
					physics_scene->steps_since_readback++;
				}
				const clock::time_point step_end = clock::now();
				const double step_ms = std::chrono::duration<double, std::milli>(step_end - step_begin).count();
				physics_scene->stat_step_ms_acc = physics_scene->stat_step_ms_acc + step_ms;
				physics_scene->stat_step_count++;
				if (step_ms > physics_scene->stat_step_ms_max)
				{
					physics_scene->stat_step_ms_max = step_ms;
				}

				next += period;
				const double lag_ms = std::chrono::duration<double, std::milli>(step_end - next).count(); // positive: behind
				physics_scene->stat_wall_lag_ms = lag_ms;
				if (lag_ms > TIMESTEP * ACCURACY * 1000.0)
				{
					physics_scene->stat_resync_count++;
					synced = false;
				}
				else if (lag_ms < 0)
				{
					std::this_thread::sleep_until(next);
				}
			}
		}

		void PhysicsScene::StartStepThread()
		{
			if (step_thread_running)
				return;
			step_thread_quit = false;
			step_thread_running = true;
			step_thread = std::thread(StepThreadMain, this);
		}

		void StepPhysicsScene(PhysicsScene& physics_scene)
		{
			for (Articulation* articulation : physics_scene.articulations)
			{
				ArticulationPreStep(*articulation);
			}
			physics_scene.contact_callback.dt = TIMESTEP;
			physics_scene.scene->simulate(TIMESTEP);
			physics_scene.scene->fetchResults(true);
			physics_scene.sim_time += TIMESTEP;
			physics_scene.step_count++;
			for (Articulation* articulation : physics_scene.articulations)
			{
				ArticulationPostStep(physics_scene, *articulation, TIMESTEP);
			}
		}
	}

	// Physics debug draw: wireframes of the shapes as PhysX holds them (the cooked geometry and the actual poses), so that what
	//	the solver collides with can be compared with what is rendered. Shapes farther than DEBUG_MAX_DRAW_DISTANCE from the camera
	//	and meshes / height fields with more than kDebugMaxTriangles triangles are skipped, like the Jolt backend does.
	//	Runs on the main thread inside the update (the stepping thread is blocked), wi::renderer::DrawLine is not thread safe
	static constexpr uint32_t kDebugMaxTriangles = 200000;
	static void DebugDrawShapes(const Scene& scene, PhysicsScene& physics_scene)
	{
		if (physics_scene.scene == nullptr)
			return;
		const PxVec3 eye = PxVec3(scene.camera.Eye.x, scene.camera.Eye.y, scene.camera.Eye.z);

		auto line = [](const PxVec3& a, const PxVec3& b, const XMFLOAT4& color) {
			wi::renderer::RenderableLine l;
			l.start = XMFLOAT3(a.x, a.y, a.z);
			l.end = XMFLOAT3(b.x, b.y, b.z);
			l.color_start = l.color_end = color;
			wi::renderer::DrawLine(l);
		};
		// circle of radius r around the pose's local axis (0 = x, 1 = y, 2 = z), centred at local c
		auto circle = [&](const PxTransform& pose, const PxVec3& c, float r, int axis, const XMFLOAT4& color, float a0 = 0, float a1 = PxTwoPi) {
			const int segments = 24;
			PxVec3 prev(0.0f);
			for (int k = 0; k <= segments; ++k)
			{
				const float a = a0 + (a1 - a0) * k / segments;
				const float u = r * std::cos(a), v = r * std::sin(a);
				PxVec3 p = c;
				if (axis == 0) { p.y += u; p.z += v; }
				else if (axis == 1) { p.z += u; p.x += v; }
				else { p.x += u; p.y += v; }
				p = pose.transform(p);
				if (k > 0)
					line(prev, p, color);
				prev = p;
			}
		};

		auto draw_shape = [&](const PxRigidActor& actor, const PxShape& shape, const XMFLOAT4& color) {
			const PxTransform pose = PxShapeExt::getGlobalPose(shape, actor);
			if ((pose.p - eye).magnitude() > DEBUG_MAX_DRAW_DISTANCE)
				return;
			const PxGeometry& geometry = shape.getGeometry();
			switch (geometry.getType())
			{
			case PxGeometryType::eBOX:
			{
				const PxVec3 h = static_cast<const PxBoxGeometry&>(geometry).halfExtents;
				PxVec3 c[8];
				for (int k = 0; k < 8; ++k)
					c[k] = pose.transform(PxVec3(k & 1 ? h.x : -h.x, k & 2 ? h.y : -h.y, k & 4 ? h.z : -h.z));
				for (int k = 0; k < 8; ++k)
				{
					for (int bit = 1; bit < 8; bit <<= 1)
					{
						if ((k & bit) == 0)
							line(c[k], c[k | bit], color);
					}
				}
			}
			break;
			case PxGeometryType::eSPHERE:
			{
				const float r = static_cast<const PxSphereGeometry&>(geometry).radius;
				for (int axis = 0; axis < 3; ++axis)
					circle(pose, PxVec3(0), r, axis, color);
			}
			break;
			case PxGeometryType::eCAPSULE:
			{
				// PhysX capsules run along the local x axis
				const PxCapsuleGeometry& capsule = static_cast<const PxCapsuleGeometry&>(geometry);
				const float r = capsule.radius, hh = capsule.halfHeight;
				circle(pose, PxVec3(hh, 0, 0), r, 0, color);
				circle(pose, PxVec3(-hh, 0, 0), r, 0, color);
				const int sides = 12; // lines along the axis, enough to read the width from any direction
				for (int k = 0; k < sides; ++k)
				{
					const float a = k * PxTwoPi / sides;
					const PxVec3 o(0, r * std::cos(a), r * std::sin(a));
					line(pose.transform(PxVec3(hh, 0, 0) + o), pose.transform(PxVec3(-hh, 0, 0) + o), color);
				}
				// end caps: half circles in the xy and xz planes
				circle(pose, PxVec3(hh, 0, 0), r, 2, color, -PxHalfPi, PxHalfPi);
				circle(pose, PxVec3(-hh, 0, 0), r, 2, color, PxHalfPi, PxHalfPi * 3);
				circle(pose, PxVec3(hh, 0, 0), r, 1, color, 0, PxPi);
				circle(pose, PxVec3(-hh, 0, 0), r, 1, color, PxPi, PxTwoPi);
			}
			break;
			case PxGeometryType::eCONVEXMESH:
			{
				const PxConvexMeshGeometry& convex = static_cast<const PxConvexMeshGeometry&>(geometry);
				const PxConvexMesh* mesh = convex.convexMesh;
				if (mesh == nullptr)
					break;
				const PxMat33 scale = convex.scale.toMat33();
				const PxVec3* vertices = mesh->getVertices();
				const PxU8* indices = mesh->getIndexBuffer();
				for (PxU32 i = 0; i < mesh->getNbPolygons(); ++i)
				{
					PxHullPolygon polygon;
					if (!mesh->getPolygonData(i, polygon))
						continue;
					for (PxU32 k = 0; k < polygon.mNbVerts; ++k)
					{
						const PxVec3 a = vertices[indices[polygon.mIndexBase + k]];
						const PxVec3 b = vertices[indices[polygon.mIndexBase + (k + 1) % polygon.mNbVerts]];
						line(pose.transform(scale * a), pose.transform(scale * b), color);
					}
				}
			}
			break;
			case PxGeometryType::eTRIANGLEMESH:
			{
				const PxTriangleMeshGeometry& tm = static_cast<const PxTriangleMeshGeometry&>(geometry);
				const PxTriangleMesh* mesh = tm.triangleMesh;
				if (mesh == nullptr || mesh->getNbTriangles() > kDebugMaxTriangles)
					break;
				const PxMat33 scale = tm.scale.toMat33();
				const PxVec3* vertices = mesh->getVertices();
				const bool wide = !(mesh->getTriangleMeshFlags() & PxTriangleMeshFlag::e16_BIT_INDICES);
				const void* indices = mesh->getTriangles();
				for (PxU32 t = 0; t < mesh->getNbTriangles(); ++t)
				{
					PxU32 v[3];
					for (int k = 0; k < 3; ++k)
						v[k] = wide ? static_cast<const PxU32*>(indices)[3 * t + k] : static_cast<const PxU16*>(indices)[3 * t + k];
					for (int k = 0; k < 3; ++k)
						line(pose.transform(scale * vertices[v[k]]), pose.transform(scale * vertices[v[(k + 1) % 3]]), color);
				}
			}
			break;
			case PxGeometryType::eHEIGHTFIELD:
			{
				const PxHeightFieldGeometry& hf = static_cast<const PxHeightFieldGeometry&>(geometry);
				const PxHeightField* field = hf.heightField;
				if (field == nullptr)
					break;
				const PxU32 rows = field->getNbRows(), columns = field->getNbColumns();
				if (rows < 2 || columns < 2 || (uint64_t)(rows - 1) * (columns - 1) * 2 > kDebugMaxTriangles)
					break;
				// sample (row, column) is at local (row * rowScale, height * heightScale, column * columnScale)
				auto vertex = [&](PxU32 r, PxU32 c) {
					return pose.transform(PxVec3(r * hf.rowScale, field->getHeight((PxReal)r, (PxReal)c) * hf.heightScale, c * hf.columnScale));
				};
				for (PxU32 r = 0; r < rows; ++r)
				{
					for (PxU32 c = 0; c < columns; ++c)
					{
						if (r + 1 < rows) line(vertex(r, c), vertex(r + 1, c), color);
						if (c + 1 < columns) line(vertex(r, c), vertex(r, c + 1), color);
					}
				}
			}
			break;
			default:
				break;
			}
		};

		auto draw_actor = [&](const PxRigidActor& actor, const XMFLOAT4& color) {
			const PxU32 count = actor.getNbShapes();
			wi::vector<PxShape*> shapes(count);
			actor.getShapes(shapes.data(), count);
			for (const PxShape* shape : shapes)
				draw_shape(actor, *shape, color);
		};

		const XMFLOAT4 color_static = XMFLOAT4(0.3f, 1.0f, 0.3f, 1);
		const XMFLOAT4 color_dynamic = XMFLOAT4(1.0f, 0.6f, 0.1f, 1);
		const XMFLOAT4 color_link = XMFLOAT4(0.2f, 0.8f, 1.0f, 1);
		const PxActorTypeFlags types = PxActorTypeFlag::eRIGID_STATIC | PxActorTypeFlag::eRIGID_DYNAMIC;
		const PxU32 actor_count = physics_scene.scene->getNbActors(types);
		wi::vector<PxActor*> actors(actor_count);
		physics_scene.scene->getActors(types, actors.data(), actor_count);
		for (PxActor* actor : actors)
		{
			if (const PxRigidActor* rigid = actor->is<PxRigidActor>())
				draw_actor(*rigid, actor->getType() == PxActorType::eRIGID_STATIC ? color_static : color_dynamic);
		}
		const PxU32 articulation_count = physics_scene.scene->getNbArticulations();
		wi::vector<PxArticulationReducedCoordinate*> articulations(articulation_count);
		physics_scene.scene->getArticulations(articulations.data(), articulation_count);
		for (PxArticulationReducedCoordinate* articulation : articulations)
		{
			const PxU32 link_count = articulation->getNbLinks();
			wi::vector<PxArticulationLink*> links(link_count);
			articulation->getLinks(links.data(), link_count);
			for (const PxArticulationLink* link : links)
				draw_actor(*link, color_link);
		}
	}

	void RunPhysicsUpdateSystem(wi::jobsystem::context& ctx, Scene& scene, float dt)
	{
		if (!IsEnabled() || dt <= 0)
			return;

		wi::jobsystem::Wait(ctx);
		scene.RunHierarchyUpdateSystem(ctx);
		wi::jobsystem::Wait(ctx);

		auto range = wi::profiler::BeginRangeCPU("Physics");

		PhysicsScene& physics_scene = GetPhysicsScene(scene);
		if (ASYNC_SIMULATION && !physics_scene.step_thread_running)
		{
			physics_scene.StartStepThread();
		}
		else if (!ASYNC_SIMULATION && physics_scene.step_thread_running)
		{
			physics_scene.StopStepThread();
		}
		UpdateScope update_scope(physics_scene);
		physics_scene.scene->setGravity(cast(scene.weather.gravity));

		// Creation. Articulation links make their own bodies (stage C), everything else lands here.
		wi::jobsystem::Dispatch(ctx, (uint32_t)scene.rigidbodies.GetCount(), dispatchGroupSize, [&scene](wi::jobsystem::JobArgs args) {

			RigidBodyPhysicsComponent& physicscomponent = scene.rigidbodies[args.jobIndex];
			const Entity entity = scene.rigidbodies.GetEntity(args.jobIndex);
			if (scene.articulation_links.Contains(entity))
				return;

			if ((physicscomponent.physicsobject == nullptr || physicscomponent.IsRefreshParametersNeeded()) && scene.transforms.Contains(entity))
			{
				physicscomponent.SetRefreshParametersNeeded(false);
				const TransformComponent* transform = scene.transforms.GetComponent(entity);
				if (transform == nullptr)
					return;
				const ObjectComponent* object = scene.objects.GetComponent(entity);
				const MeshComponent* mesh = object != nullptr ? scene.meshes.GetComponent(object->meshID) : nullptr;
				AddRigidBody(scene, entity, physicscomponent, *transform, mesh);
			}
		});
		wi::jobsystem::Wait(ctx);

		if (scene.softbodies.GetCount() > 0)
			unimplemented("soft bodies");
		if (scene.humanoids.GetCount() > 0)
			unimplemented("ragdolls");
		if (scene.constraints.GetCount() > 0)
			unimplemented("PhysicsConstraintComponent");

		// Articulation creation (serial: the number of articulations is expected to be small)
		{
			wi::vector<Entity> link_entities;
			wi::vector<int> parent_indices;
			for (size_t i = 0; i < scene.articulations.GetCount(); ++i)
			{
				ArticulationComponent& physicscomponent = scene.articulations[i];
				const Entity root_entity = scene.articulations.GetEntity(i);
				bool recreate = physicscomponent.physicsobject == nullptr;
				if (!recreate)
				{
					// Structural changes (links removed or added) need a rebuild: PhysX articulations are fixed
					//	once they are in a scene
					const Articulation& articulation = GetArticulation(physicscomponent);
					for (auto& link : articulation.links)
					{
						if (!scene.articulation_links.Contains(link->entity))
						{
							recreate = true;
							break;
						}
					}
					if (!recreate && CollectArticulationLinks(scene, root_entity, link_entities, parent_indices)
						&& link_entities.size() != articulation.links.size())
					{
						recreate = true;
					}
				}
				if (recreate)
				{
					AddArticulation(scene, root_entity, physicscomponent);
				}
			}
		}

		// Articulation drive updates (every frame)
		for (size_t i = 0; i < scene.articulations.GetCount(); ++i)
		{
			ArticulationComponent& physicscomponent = scene.articulations[i];
			if (physicscomponent.physicsobject == nullptr)
				continue;
			Articulation& articulation = GetArticulation(physicscomponent);
			const bool refresh_all = physicscomponent.IsRefreshParametersNeeded();
			physicscomponent.SetRefreshParametersNeeded(false);

			for (size_t l = 0; l < articulation.links.size(); ++l)
			{
				Articulation::Link& link = *articulation.links[l];
				ArticulationLinkComponent* linkcomponent = scene.articulation_links.GetComponent(link.entity);
				if (linkcomponent == nullptr)
					continue;

				if (articulation.external_drive && l < articulation.commands.size())
				{
					// The step callback owns the drive; write the last command back for display
					const wi::physics::ArticulationLinkCommand& cmd = articulation.commands[l];
					linkcomponent->drive.target_position = cmd.target_position;
					linkcomponent->drive.target_velocity = cmd.target_velocity;
					linkcomponent->drive.stiffness = cmd.stiffness;
					linkcomponent->drive.damping = cmd.damping;
					linkcomponent->drive.max_force = cmd.max_force;
					linkcomponent->drive.feedforward_force = cmd.feedforward_force;
				}
				else
				{
					wi::physics::ArticulationLinkCommand cmd;
					cmd.target_position = linkcomponent->drive.target_position;
					cmd.target_velocity = linkcomponent->drive.target_velocity;
					cmd.stiffness = linkcomponent->drive.stiffness;
					cmd.damping = linkcomponent->drive.damping;
					cmd.max_force = linkcomponent->drive.max_force;
					cmd.feedforward_force = linkcomponent->drive.feedforward_force;
					ApplyLinkDrive(link, cmd);
				}

				if (link.joint == nullptr || !link.has_dof)
					continue;
				link.joint->setFrictionParams(PxArticulationAxis::eTWIST,
					PxJointFrictionParams(std::max(0.0f, linkcomponent->joint_friction), std::max(0.0f, linkcomponent->joint_friction), 0.0f));

				if (refresh_all || linkcomponent->IsRefreshParametersNeeded())
				{
					linkcomponent->SetRefreshParametersNeeded(false);
					const bool has_limits = linkcomponent->limit_min > -XM_PI || linkcomponent->limit_max < XM_PI;
					if (has_limits)
					{
						link.joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eLIMITED);
						link.joint->setLimitParams(PxArticulationAxis::eTWIST, PxArticulationLimit(
							linkcomponent->limit_min - link.angle_offset, linkcomponent->limit_max - link.angle_offset));
					}
					else
					{
						link.joint->setMotion(PxArticulationAxis::eTWIST, PxArticulationMotion::eFREE);
					}
				}
			}
		}

		// Property updates and system-to-physics feedback for anything the simulation does not own.
		for (size_t i = 0; i < scene.rigidbodies.GetCount(); ++i)
		{
			RigidBodyPhysicsComponent& physicscomponent = scene.rigidbodies[i];
			const Entity entity = scene.rigidbodies.GetEntity(i);
			if (physicscomponent.physicsobject == nullptr)
				continue;
			RigidBody& physicsobject = GetRigidBody(physicscomponent);
			if (physicsobject.actor == nullptr)
				continue;

			if (physicsobject.material != nullptr &&
				(physicsobject.friction != physicscomponent.friction || physicsobject.restitution != physicscomponent.restitution))
			{
				physicsobject.friction = physicscomponent.friction;
				physicsobject.restitution = physicscomponent.restitution;
				physicsobject.material->setStaticFriction(physicscomponent.friction);
				physicsobject.material->setDynamicFriction(physicscomponent.friction);
				physicsobject.material->setRestitution(physicscomponent.restitution);
			}

			// A change of motion type means a different actor class in PhysX, so ask for a rebuild instead of
			//	mutating in place. Jolt can switch a body's motion type; this is the honest equivalent.
			const bool wants_dynamic = physicscomponent.mass > 0 && !physicsobject.shape.static_only;
			const bool wants_kinematic = wants_dynamic && physicscomponent.IsKinematic();
			if (wants_dynamic != physicsobject.is_dynamic)
			{
				physicscomponent.SetRefreshParametersNeeded(true);
				continue;
			}
			if (physicsobject.is_dynamic && wants_kinematic != physicsobject.is_kinematic)
			{
				physicsobject.is_kinematic = wants_kinematic;
				static_cast<PxRigidDynamic*>(physicsobject.actor)->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, wants_kinematic);
			}

			TransformComponent* transform = scene.transforms.GetComponent(entity);
			if (transform == nullptr)
				continue;

			if (physicsobject.is_dynamic && !physicsobject.is_kinematic)
			{
				transform->MatrixTransform(physicsobject.parentMatrix);
			}

			if (physics_scene.activate_all_rigid_bodies && physicsobject.is_dynamic && !physicsobject.is_kinematic)
			{
				static_cast<PxRigidDynamic*>(physicsobject.actor)->wakeUp();
			}

			if (physicsobject.teleporting)
			{
				physicsobject.teleporting = false;
				continue;
			}

			const PxVec3 position = cast(transform->GetPosition());
			const PxQuat rotation = cast(transform->GetRotation()).getNormalized();
			const PxTransform pose(position + physicsobject.local_offset, rotation);

			if (!IsSimulationEnabled())
			{
				physicsobject.prev_position = position;
				physicsobject.prev_rotation = rotation;
				physicsobject.actor->setGlobalPose(pose);
			}
			else if (physicsobject.is_kinematic)
			{
				static_cast<PxRigidDynamic*>(physicsobject.actor)->setKinematicTarget(pose);
			}
			else if (!physicsobject.is_dynamic || !physicsobject.was_active_prev_frame)
			{
				physicsobject.actor->setGlobalPose(pose);
			}
		}
		physics_scene.activate_all_rigid_bodies = false;

		// Simulation steps
		physics_scene.steps_last_frame = 0;
		if (physics_scene.step_thread_running)
		{
			// The stepping thread did them; nothing to interpolate between, the readback takes the latest state
			physics_scene.steps_last_frame = (uint32_t)physics_scene.steps_since_readback.exchange(0);
			physics_scene.accumulator = 0;
			physics_scene.alpha = 0;
		}
		else if (IsSimulationEnabled())
		{
			physics_scene.accumulator = std::min(physics_scene.accumulator + dt, TIMESTEP * ACCURACY);
			while (physics_scene.accumulator >= TIMESTEP)
			{
				const float next_accumulator = physics_scene.accumulator - TIMESTEP;
				if (IsInterpolationEnabled() && next_accumulator < TIMESTEP)
				{
					// Last step of the frame: remember where things were, so the render frame can interpolate
					for (size_t i = 0; i < scene.rigidbodies.GetCount(); ++i)
					{
						RigidBodyPhysicsComponent& physicscomponent = scene.rigidbodies[i];
						if (physicscomponent.physicsobject == nullptr)
							continue;
						RigidBody& rb = GetRigidBody(physicscomponent);
						if (rb.actor == nullptr)
							continue;
						const PxTransform pose = rb.actor->getGlobalPose();
						rb.prev_position = pose.p - rb.local_offset;
						rb.prev_rotation = pose.q;
					}
				}
				StepPhysicsScene(physics_scene);
				physics_scene.accumulator = next_accumulator;
				physics_scene.steps_last_frame++;
			}
			physics_scene.alpha = physics_scene.accumulator / TIMESTEP;
		}

		// Physics back to the system
		for (size_t i = 0; i < scene.rigidbodies.GetCount(); ++i)
		{
			RigidBodyPhysicsComponent& physicscomponent = scene.rigidbodies[i];
			if (physicscomponent.physicsobject == nullptr)
				continue;
			RigidBody& physicsobject = GetRigidBody(physicscomponent);
			if (physicsobject.actor == nullptr || !physicsobject.is_dynamic || physicsobject.is_kinematic)
				continue;

			PxRigidDynamic* body = static_cast<PxRigidDynamic*>(physicsobject.actor);
			physicsobject.was_active_prev_frame = !body->isSleeping();

			const Entity entity = scene.rigidbodies.GetEntity(i);
			TransformComponent* transform = scene.transforms.GetComponent(entity);
			if (transform == nullptr)
				continue;

			const PxTransform pose = body->getGlobalPose();
			PxVec3 position = pose.p - physicsobject.local_offset;
			PxQuat rotation = pose.q;

			if (IsInterpolationEnabled())
			{
				position = position * physics_scene.alpha + physicsobject.prev_position * (1 - physics_scene.alpha);
				rotation = PxSlerp(physics_scene.alpha, physicsobject.prev_rotation, rotation);
			}

			transform->translation_local = cast(position);
			transform->rotation_local = cast(rotation);
			transform->MatrixTransform(physicsobject.parentMatrixInverse);
		}

		// Articulation feedback: link transforms and the state measured in ArticulationPostStep
		for (size_t i = 0; i < scene.articulations.GetCount(); ++i)
		{
			ArticulationComponent& physicscomponent = scene.articulations[i];
			if (physicscomponent.physicsobject == nullptr)
				continue;
			Articulation& articulation = GetArticulation(physicscomponent);
			for (auto& link_ptr : articulation.links)
			{
				Articulation::Link& link = *link_ptr;
				if (link.link == nullptr)
					continue;
				TransformComponent* transform = scene.transforms.GetComponent(link.entity);
				if (transform != nullptr)
				{
					const PxTransform pose = link.link->getGlobalPose();
					transform->translation_local = cast(pose.p);
					transform->rotation_local = cast(pose.q.getNormalized());
					transform->MatrixTransform(link.parentMatrixInverse);
				}
				ArticulationLinkComponent* linkcomponent = scene.articulation_links.GetComponent(link.entity);
				if (linkcomponent == nullptr)
					continue;
				linkcomponent->contact_sensors = link.contact_snapshot;
				linkcomponent->position = link.position;
				linkcomponent->velocity = link.velocity;
				linkcomponent->force = link.motor_force + link.limit_force;
				linkcomponent->joint_force = cast(link.joint_force);
				linkcomponent->external_force = cast(link.external_force);
			}
			physicscomponent.root_linear_velocity = cast(articulation.root_linear_velocity);
			physicscomponent.root_angular_velocity = cast(articulation.root_angular_velocity);
			physicscomponent.root_linear_acceleration = cast(articulation.root_linear_acceleration);
		}

		if (DEBUGDRAW_ENABLED)
		{
			DebugDrawShapes(scene, physics_scene);
		}

		wi::profiler::EndRange(range);
	}

	// ---- Global settings -------------------------------------------------------------------------------------

	bool IsEnabled() { return ENABLED; }
	void SetEnabled(bool value) { ENABLED = value; }

	bool IsSimulationEnabled() { return ENABLED && SIMULATION_ENABLED; }
	void SetSimulationEnabled(bool value) { SIMULATION_ENABLED = value; }

	bool IsInterpolationEnabled() { return INTERPOLATION && !ASYNC_SIMULATION; }
	void SetInterpolationEnabled(bool value) { INTERPOLATION = value; }

	bool IsDebugDrawEnabled() { return DEBUGDRAW_ENABLED; }
	void SetDebugDrawEnabled(bool value) { DEBUGDRAW_ENABLED = value; }

	void SetConstraintDebugSize(float value) { CONSTRAINT_DEBUGSIZE = value; }
	float GetConstraintDebugSize() { return CONSTRAINT_DEBUGSIZE; }

	void SetDebugDrawMaxDistance(float value) { DEBUG_MAX_DRAW_DISTANCE = value; }
	float GetDebugDrawMaxDistance() { return DEBUG_MAX_DRAW_DISTANCE; }

	int GetAccuracy() { return ACCURACY; }
	void SetAccuracy(int value) { ACCURACY = value; }

	float GetFrameRate() { return 1.0f / TIMESTEP; }
	void SetFrameRate(float value) { TIMESTEP = 1.0f / value; }

	float GetCharacterCollisionTolerance() { return CHARACTER_COLLISION_TOLERANCE; }
	void SetCharacterCollisionTolerance(float value) { CHARACTER_COLLISION_TOLERANCE = value; }

	void SetSolverIterations(int velocity_iterations, int position_iterations)
	{
		SOLVER_VELOCITY_ITERATIONS = std::max(0, velocity_iterations); // PhysX allows 0 velocity iterations, Jolt does not
		SOLVER_POSITION_ITERATIONS = std::max(1, position_iterations);
	}
	int GetSolverVelocityIterations() { return SOLVER_VELOCITY_ITERATIONS; }
	int GetSolverPositionIterations() { return SOLVER_POSITION_ITERATIONS; }

	void SetAsyncSimulationEnabled(bool value) { ASYNC_SIMULATION = value; }
	bool IsAsyncSimulationEnabled() { return ASYNC_SIMULATION; }
	AsyncSimulationStats GetAsyncSimulationStats(Scene& scene)
	{
		AsyncSimulationStats stats;
		if (scene.physics_scene == nullptr)
			return stats;
		PhysicsScene& physics_scene = *(PhysicsScene*)scene.physics_scene.get();
		stats.sim_time = physics_scene.sim_time;
		stats.steps = physics_scene.step_count;
		stats.wall_lag_ms = physics_scene.stat_wall_lag_ms;
		const uint32_t count = physics_scene.stat_step_count.exchange(0);
		const double acc = physics_scene.stat_step_ms_acc.exchange(0);
		stats.step_time_ms_avg = count > 0 ? acc / count : 0;
		stats.step_time_ms_max = physics_scene.stat_step_ms_max.exchange(0);
		stats.resync_count = physics_scene.stat_resync_count;
		return stats;
	}

	// ---- Rigid body access -----------------------------------------------------------------------------------

	namespace
	{
		PxRigidDynamic* GetDynamic(RigidBodyPhysicsComponent& physicscomponent)
		{
			if (physicscomponent.physicsobject == nullptr)
				return nullptr;
			RigidBody& physicsobject = GetRigidBody(physicscomponent);
			if (physicsobject.actor == nullptr || !physicsobject.is_dynamic)
				return nullptr;
			return static_cast<PxRigidDynamic*>(physicsobject.actor);
		}
	}

	void SetPosition(RigidBodyPhysicsComponent& physicscomponent, const XMFLOAT3& position)
	{
		if (physicscomponent.physicsobject == nullptr)
			return;
		RigidBody& physicsobject = GetRigidBody(physicscomponent);
		if (physicsobject.actor == nullptr)
			return;
		StepLock step_lock(*(PhysicsScene*)physicsobject.physics_scene.get());
		const PxQuat rotation = physicsobject.actor->getGlobalPose().q;
		physicsobject.actor->setGlobalPose(PxTransform(cast(position) + physicsobject.local_offset, rotation));
		physicsobject.prev_position = cast(position);
		physicsobject.teleporting = true;
	}
	void SetPositionAndRotation(RigidBodyPhysicsComponent& physicscomponent, const XMFLOAT3& position, const XMFLOAT4& rotation)
	{
		if (physicscomponent.physicsobject == nullptr)
			return;
		RigidBody& physicsobject = GetRigidBody(physicscomponent);
		if (physicsobject.actor == nullptr)
			return;
		StepLock step_lock(*(PhysicsScene*)physicsobject.physics_scene.get());
		physicsobject.actor->setGlobalPose(PxTransform(cast(position) + physicsobject.local_offset, cast(rotation).getNormalized()));
		physicsobject.prev_position = cast(position);
		physicsobject.prev_rotation = cast(rotation).getNormalized();
		physicsobject.teleporting = true;
	}
	void SetLinearVelocity(RigidBodyPhysicsComponent& physicscomponent, const XMFLOAT3& velocity)
	{
		PxRigidDynamic* body = GetDynamic(physicscomponent);
		if (body != nullptr)
			body->setLinearVelocity(cast(velocity));
	}
	void SetAngularVelocity(RigidBodyPhysicsComponent& physicscomponent, const XMFLOAT3& velocity)
	{
		PxRigidDynamic* body = GetDynamic(physicscomponent);
		if (body != nullptr)
			body->setAngularVelocity(cast(velocity));
	}
	XMFLOAT3 GetVelocity(RigidBodyPhysicsComponent& physicscomponent)
	{
		PxRigidDynamic* body = GetDynamic(physicscomponent);
		return body != nullptr ? cast(body->getLinearVelocity()) : XMFLOAT3(0, 0, 0);
	}
	XMFLOAT3 GetPosition(RigidBodyPhysicsComponent& physicscomponent)
	{
		if (physicscomponent.physicsobject == nullptr)
			return XMFLOAT3(0, 0, 0);
		RigidBody& physicsobject = GetRigidBody(physicscomponent);
		if (physicsobject.actor == nullptr)
			return XMFLOAT3(0, 0, 0);
		return cast(physicsobject.actor->getGlobalPose().p - physicsobject.local_offset);
	}
	XMFLOAT4 GetRotation(RigidBodyPhysicsComponent& physicscomponent)
	{
		if (physicscomponent.physicsobject == nullptr)
			return XMFLOAT4(0, 0, 0, 1);
		RigidBody& physicsobject = GetRigidBody(physicscomponent);
		if (physicsobject.actor == nullptr)
			return XMFLOAT4(0, 0, 0, 1);
		return cast(physicsobject.actor->getGlobalPose().q);
	}

	void ApplyForce(RigidBodyPhysicsComponent& physicscomponent, const XMFLOAT3& force)
	{
		PxRigidDynamic* body = GetDynamic(physicscomponent);
		if (body != nullptr)
			body->addForce(cast(force), PxForceMode::eFORCE);
	}
	void ApplyForceAt(RigidBodyPhysicsComponent& physicscomponent, const XMFLOAT3& force, const XMFLOAT3& at, bool at_local)
	{
		PxRigidDynamic* body = GetDynamic(physicscomponent);
		if (body == nullptr)
			return;
		if (at_local)
			PxRigidBodyExt::addForceAtLocalPos(*body, cast(force), cast(at), PxForceMode::eFORCE);
		else
			PxRigidBodyExt::addForceAtPos(*body, cast(force), cast(at), PxForceMode::eFORCE);
	}
	void ApplyImpulse(RigidBodyPhysicsComponent& physicscomponent, const XMFLOAT3& impulse)
	{
		PxRigidDynamic* body = GetDynamic(physicscomponent);
		if (body != nullptr)
			body->addForce(cast(impulse), PxForceMode::eIMPULSE);
	}
	void ApplyImpulseAt(RigidBodyPhysicsComponent& physicscomponent, const XMFLOAT3& impulse, const XMFLOAT3& at, bool at_local)
	{
		PxRigidDynamic* body = GetDynamic(physicscomponent);
		if (body == nullptr)
			return;
		if (at_local)
			PxRigidBodyExt::addForceAtLocalPos(*body, cast(impulse), cast(at), PxForceMode::eIMPULSE);
		else
			PxRigidBodyExt::addForceAtPos(*body, cast(impulse), cast(at), PxForceMode::eIMPULSE);
	}
	void ApplyTorque(RigidBodyPhysicsComponent& physicscomponent, const XMFLOAT3& torque)
	{
		PxRigidDynamic* body = GetDynamic(physicscomponent);
		if (body != nullptr)
			body->addTorque(cast(torque), PxForceMode::eFORCE);
	}

	void SetActivationState(RigidBodyPhysicsComponent& physicscomponent, ActivationState state)
	{
		PxRigidDynamic* body = GetDynamic(physicscomponent);
		if (body == nullptr || body->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC))
			return;
		if (state == ActivationState::Active)
			body->wakeUp();
		else
			body->putToSleep();
	}
	void SetActivationState(SoftBodyPhysicsComponent& physicscomponent, ActivationState state)
	{
		unimplemented("soft bodies");
	}
	void ActivateAllRigidBodies(Scene& scene)
	{
		if (scene.physics_scene == nullptr)
			return;
		((PhysicsScene*)scene.physics_scene.get())->activate_all_rigid_bodies = true;
	}
	void OptimizeBroadPhase(Scene& scene)
	{
		// PhysX maintains its broadphase incrementally, there is nothing to request here.
	}
	void ResetPhysicsObjects(Scene& scene)
	{
		for (size_t i = 0; i < scene.rigidbodies.GetCount(); ++i)
		{
			RigidBodyPhysicsComponent& physicscomponent = scene.rigidbodies[i];
			if (physicscomponent.physicsobject == nullptr)
				continue;
			RigidBody& physicsobject = GetRigidBody(physicscomponent);
			if (physicsobject.actor == nullptr || physicsobject.physics_scene == nullptr)
				continue;
			StepLock step_lock(*(PhysicsScene*)physicsobject.physics_scene.get());
			physicsobject.actor->setGlobalPose(PxTransform(physicsobject.initial_position + physicsobject.local_offset, physicsobject.initial_rotation));
			physicsobject.prev_position = physicsobject.initial_position;
			physicsobject.prev_rotation = physicsobject.initial_rotation;
			if (physicsobject.is_dynamic && !physicsobject.is_kinematic)
			{
				PxRigidDynamic* body = static_cast<PxRigidDynamic*>(physicsobject.actor);
				body->setLinearVelocity(PxVec3(0.0f));
				body->setAngularVelocity(PxVec3(0.0f));
			}
		}
	}
	void SetGhostMode(RigidBodyPhysicsComponent& physicscomponent, bool value)
	{
		if (physicscomponent.physicsobject == nullptr)
			return;
		RigidBody& physicsobject = GetRigidBody(physicscomponent);
		if (physicsobject.actor == nullptr || physicsobject.physics_scene == nullptr)
			return;
		StepLock step_lock(*(PhysicsScene*)physicsobject.physics_scene.get());
		PxShape* shape = nullptr;
		if (physicsobject.actor->getShapes(&shape, 1) == 1 && shape != nullptr)
		{
			shape->setFlag(PxShapeFlag::eSIMULATION_SHAPE, !value);
		}
	}

	RayIntersectionResult Intersects(const Scene& scene, wi::primitive::Ray ray)
	{
		RayIntersectionResult result;
		if (scene.physics_scene == nullptr)
			return result;
		const PhysicsScene& physics_scene = *(PhysicsScene*)scene.physics_scene.get();
		if (physics_scene.scene == nullptr)
			return result;

		PxVec3 direction = cast(ray.direction);
		if (direction.normalizeSafe() <= 0)
			return result;
		const float t_min = std::max(0.0f, ray.TMin);
		const float distance = ray.TMax - t_min;
		if (distance <= 0)
			return result;
		const PxVec3 origin = cast(ray.origin) + direction * t_min;

		StepLock step_lock(physics_scene);
		PxRaycastBuffer hit;
		if (!physics_scene.scene->raycast(origin, direction, distance, hit) || !hit.hasBlock)
			return result;

		const ActorUserData* owner = hit.block.actor != nullptr ? (const ActorUserData*)hit.block.actor->userData : nullptr;
		if (owner == nullptr)
			return result;
		result.entity = owner->entity;
		result.position = cast(hit.block.position);
		result.normal = cast(hit.block.normal);
		result.physicsobject = owner;
		const PxTransform inv = hit.block.actor->getGlobalPose().getInverse();
		result.position_local = cast(inv.transform(hit.block.position));
		return result;
	}

	// ---- Not implemented here --------------------------------------------------------------------------------
	// The robot simulator does not use these, and the Editor is built with WICKED_EDITOR=OFF for the PhysX
	//	configuration so their UI never appears. They warn once and do nothing.

	void SetArticulationStepCallback(Scene& scene, Entity root, const ArticulationStepCallback& callback)
	{
		PhysicsScene& physics_scene = GetPhysicsScene(scene);
		StepLock step_lock(physics_scene);
		if (callback)
		{
			physics_scene.articulation_callbacks[root] = callback;
		}
		else
		{
			physics_scene.articulation_callbacks.erase(root);
		}
	}

	void PickDrag(const Scene& scene, wi::primitive::Ray ray, PickDragOperation& op, ConstraintType constraint_type, float break_distance)
	{
		unimplemented("PickDrag");
	}

	XMFLOAT3 GetCharacterGroundPosition(RigidBodyPhysicsComponent& physicscomponent)
	{
		unimplemented("character controller");
		return XMFLOAT3(0, 0, 0);
	}
	XMFLOAT3 GetCharacterGroundNormal(RigidBodyPhysicsComponent& physicscomponent)
	{
		unimplemented("character controller");
		return XMFLOAT3(0, 1, 0);
	}
	XMFLOAT3 GetCharacterGroundVelocity(RigidBodyPhysicsComponent& physicscomponent)
	{
		unimplemented("character controller");
		return XMFLOAT3(0, 0, 0);
	}
	bool IsCharacterGroundSupported(RigidBodyPhysicsComponent& physicscomponent)
	{
		unimplemented("character controller");
		return false;
	}
	CharacterGroundStates GetCharacterGroundState(RigidBodyPhysicsComponent& physicscomponent)
	{
		unimplemented("character controller");
		return CharacterGroundStates::InAir;
	}
	bool ChangeCharacterShape(RigidBodyPhysicsComponent& physicscomponent, const RigidBodyPhysicsComponent::CapsuleParams& capsule)
	{
		unimplemented("character controller");
		return false;
	}
	void MoveCharacter(RigidBodyPhysicsComponent& physicscomponent, const XMFLOAT3& movement_direction, float movement_speed, float jump, bool controlMovementDuringJump)
	{
		unimplemented("character controller");
	}

	void DriveVehicle(RigidBodyPhysicsComponent& physicscomponent, float forward, float right, float brake, float handbrake)
	{
		unimplemented("vehicles");
	}
	float GetVehicleForwardVelocity(RigidBodyPhysicsComponent& physicscomponent)
	{
		unimplemented("vehicles");
		return 0;
	}
	void OverrideWehicleWheelTransforms(Scene& scene)
	{
		unimplemented("vehicles");
	}

	XMFLOAT3 GetSoftBodyNodePosition(SoftBodyPhysicsComponent& physicscomponent, uint32_t physicsIndex)
	{
		// PhysX 5 deformable volumes and surfaces are GPU only, and this build is the CPU pipeline
		unimplemented("soft bodies (PhysX 5 deformables are GPU only)");
		return XMFLOAT3(0, 0, 0);
	}

	void ApplyImpulse(HumanoidComponent& humanoid, HumanoidComponent::HumanoidBone bone, const XMFLOAT3& impulse)
	{
		unimplemented("ragdolls");
	}
	void ApplyImpulseAt(HumanoidComponent& humanoid, HumanoidComponent::HumanoidBone bone, const XMFLOAT3& impulse, const XMFLOAT3& at, bool at_local)
	{
		unimplemented("ragdolls");
	}
	void SetGhostMode(HumanoidComponent& humanoid, bool value)
	{
		unimplemented("ragdolls");
	}
	void SetRagdollGhostMode(HumanoidComponent& humanoid, bool value)
	{
		unimplemented("ragdolls");
	}

	bool IsConstraintBroken(const PhysicsConstraintComponent& physicscomponent)
	{
		unimplemented("PhysicsConstraintComponent");
		return false;
	}
	void SetConstraintBroken(PhysicsConstraintComponent& physicscomponent, bool broken)
	{
		unimplemented("PhysicsConstraintComponent");
	}
}

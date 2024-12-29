import mujoco_py
model = mujoco_py.load_model_from_path("/home/hubert/.mujoco/mujoco210/model/humanoid.xml")
sim = mujoco_py.MjSim(model)
sim.step()
print(sim.data.qpos)

import wave_visualizer as vis
import wave_simulation as sim
import cv2
import torch


def load_scene_from_image(simulator, scene_image, source_brightness_scale=1.0):
    """
    Load source from an image description.
    """
    # Convert to torch.Tensor and ensure dtype and device consistency
    scene_image_torch = torch.from_numpy(scene_image).float() / 255.0  # Normalize and convert
    scene_image_torch = scene_image_torch.to(simulator.device)  # Move to the same device as simulator

    # Set refractive index field
    simulator.set_refractive_index_field(scene_image_torch[:, :, 0] * 1.5)  # Assuming max value represents 1.5 index

    # Set absorber field
    simulator.set_dampening_field(1.0 - scene_image_torch[:, :, 2], 48)

    # Set sources - Frequency set to green channel's value normalized and scaled
    sources_pos = (scene_image_torch[:, :, 1] > 0.0).nonzero(as_tuple=False)
    sources = torch.zeros((sources_pos.shape[0], 5), device=simulator.device)
    sources[:, :2] = sources_pos.float()
    sources[:, 2] = 0  # Phase
    sources[:, 3] = 1.0 * source_brightness_scale  # Amplitude
    sources[:, 4] = scene_image_torch[sources_pos[:, 0], sources_pos[:, 1], 1] * 0.5  # Frequency from green channel

    simulator.set_sources(sources.cpu().numpy())  # Assuming set_sources expects numpy array


def simulate(scene_image_fn, num_iterations, simulation_steps_per_frame, write_videos, field_colormap,
             intensity_colormap):
    # Load scene image and convert to RGB
    scene_image = cv2.cvtColor(cv2.imread(scene_image_fn), cv2.COLOR_BGR2RGB)

    # Create simulator and visualizer objects
    simulator = sim.WaveSimulator2D(scene_image.shape[1], scene_image.shape[0])
    visualizer = vis.WaveVisualizer(field_colormap=field_colormap, intensity_colormap=intensity_colormap)

    # Load scene from image file
    load_scene_from_image(simulator, scene_image)

    # Create video writers if requested
    if write_videos:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer1 = cv2.VideoWriter('simulation_field.mp4', fourcc, 60,
                                        (scene_image.shape[1], scene_image.shape[0]))
        video_writer2 = cv2.VideoWriter('simulation_intensity.mp4', fourcc, 60,
                                        (scene_image.shape[1], scene_image.shape[0]))

    # Simulation loop
    for i in range(num_iterations):
        simulator.update_sources()
        simulator.update_field()
        visualizer.update(simulator)

        if i % simulation_steps_per_frame == 0:
            frame_int = visualizer.render_intensity(1.0)
            frame_field = visualizer.render_field(0.7)

            cv2.imshow("Wave Simulation - Field", frame_field)
            cv2.imshow("Wave Simulation - Intensity", frame_int)
            cv2.waitKey(1)

            if write_videos:
                video_writer1.write(frame_field)
                video_writer2.write(frame_int)


if __name__ == "__main__":
    simulate("../example_scenes/scene_optical_fibers.png",
             10000,
             simulation_steps_per_frame=4,
             write_videos=False,
             field_colormap=vis.get_colormap_lut('RdBu', invert=True),
             intensity_colormap=vis.get_colormap_lut('afmhot', invert=False, black_level=0.1))

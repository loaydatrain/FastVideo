from fastvideo import VideoGenerator


def main():
    # Create a video generator with a pre-trained model
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_gpus=2,  # Adjust based on your hardware
    )

    # Define a prompt for your video
    prompt = "A man with wide shoulders sits in front of his computer screen, with a beautiful sunset through the window."

    # Generate the video
    video = generator.generate_video(
        prompt,
        return_frames=
        True,  # Also return frames from this call (defaults to False)
        output_path="my_videos/",  # Controls where videos are saved
        save_video=True)


if __name__ == '__main__':
    main()

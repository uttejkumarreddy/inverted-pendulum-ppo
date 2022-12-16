from BasePPOAgent import BasePPOAgent
import cv2
import numpy as np

class BasePPOAgentWithImages(BasePPOAgent):
    def __init__(self, actor, critic):
        super(BasePPOAgentWithImages, self).__init__(actor, critic)

        # Image configurations
        self.crop_proportions = (0.4, 0.0, 1.0, 1.0)
        self.crop_dim = None
        self.img_dim = (64, 64)

        self.images = []

    # Apply action and get observation from environment
    def reset_env(self):
        state = self.env.reset()

        # Store images
        img = self.env.render(mode='rgb_array')
        self.images.append(img)

        return state

    def apply_action(self, action):
        obs, reward, done, info = self.env.step([action])
        
        # Store images
        img = self.env.render(mode='rgb_array')
        self.images.append(img)

        return obs, reward, done, info

    def update_networks(self):
        batch_state, batch_action, batch_reward, batch_obs, batch_rtg = zip(*self.replay_buffer.buffer)
        
        # Calculate angular velocites based on the images collected
        angular_velocities = self.calculate_angular_velocities()
        print(angular_velocities)
        
        actor_loss = self.calculate_actor_loss(batch_state, batch_action, batch_reward, batch_obs, batch_rtg)
        critic_loss = self.calculate_critic_loss(batch_state, batch_action, batch_reward, batch_obs, batch_rtg)

        # Update gradients
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Empty images array
        self.images = []

        return { 'actor_loss': actor_loss, 'critic_loss': critic_loss }

    def calculate_angular_velocities(self):
        # Pre-processing
        image = self.images[0]
        crop_dim = (
            int(self.crop_proportions[0] * image.shape[0]),
            int(self.crop_proportions[1] * image.shape[1]),
            int(self.crop_proportions[2] * image.shape[0]),
            int(self.crop_proportions[3] * image.shape[1])
        )

        for image in self.images:
            # Crop the image for focus
            image = image[crop_dim[0]: crop_dim[2], crop_dim[1]: crop_dim[3], :]
            image = cv2.resize(image, dsize=self.img_dim, interpolation=cv2.INTER_CUBIC)

            # Convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Once all images are pre-processed, detect the position of the pendulum
        # in each image and calculate angular velocities
        xy = []
        for image in self.images:
            # Detecting xy
            image = np.float32(image)
            corners = cv2.cornerHarris(image, 2, 3, 0.04)
            corner = np.argmax(corners)
            x, y = np.unravel_index(corners, corners.shape)
            xy.append((x, y))

        theta = []
        for i in range(1, len(xy)):
            # Compute difference in positions between single timestep
            dx = xy[i][0] - xy[i - 1][0]
            dy = xy[i][1] - xy[i - 1][1]
            dt = 1

            # Assuming pivot positions
            x_pivot, y_pivot = 150, 150
            angle = np.arctan2(dy, dx) - np.arctan2(y_pivot - dy, x_pivot - dx)

            # Considering direction of the pendulum
            if dx < 0:
                angle *= -1
            
            theta.append(angle/dt)

        return theta





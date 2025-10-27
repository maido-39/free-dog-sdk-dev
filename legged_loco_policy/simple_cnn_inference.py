"""
Simple CNN Inference for ActorDepthCNN
Isaac Lab/Sim 종속성 없이 순수 torch만 사용하여 구현
"""

import torch
import torch.nn as nn


def get_activation(act_name):
    """Activation function factory"""
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


class DepthOnlyFCBackbone(nn.Module):
    """Depth image processing backbone"""
    def __init__(self, output_dim, hidden_dim, activation, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        self.output_dim = output_dim
        self.image_compression = nn.Sequential(
            # [1, 24, 32]
            nn.Conv2d(in_channels=self.num_frames, out_channels=16, kernel_size=5),
            # [16, 20, 28]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [16, 10, 14]
            activation,
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            # [32, 8, 12]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 4, 6]
            activation,
            nn.Flatten(),
            
            nn.Linear(32 * 4 * 6, hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim),
            activation
        )

    def forward(self, images: torch.Tensor):
        latent = self.image_compression(images.unsqueeze(1))
        return latent


class ActorDepthCNN(nn.Module):
    """Actor network with depth CNN"""
    def __init__(self, 
                 num_obs_proprio, 
                 obs_depth_shape, 
                 num_actions,
                 activation,
                 hidden_dims=[256, 256, 128], 
        ):
        super().__init__()

        self.prop_mlp = nn.Sequential(
            nn.Linear(num_obs_proprio, hidden_dims[0]),
            activation,
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            activation,
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            activation,
        )
        self.depth_backbone = DepthOnlyFCBackbone(
            output_dim=hidden_dims[2],
            hidden_dim=hidden_dims[1],
            activation=activation,
            num_frames=1,
        )

        self.action_head = nn.Linear(2 * hidden_dims[2], num_actions)

        self.num_obs_proprio = num_obs_proprio
        self.obs_depth_shape = obs_depth_shape
    
    def forward(self, x):
        prop_input = x[..., :self.num_obs_proprio]
        prop_latent = self.prop_mlp(prop_input)

        depth_input = x[..., self.num_obs_proprio:]
        ori_shape = depth_input.shape
        depth_input = depth_input.reshape(-1, *self.obs_depth_shape)
        depth_latent = self.depth_backbone(depth_input)

        actions = self.action_head(torch.cat((prop_latent, depth_latent), dim=-1))
        return actions


def load_policy(model_path):
    """
    Load actor network from model.pt checkpoint
    
    Args:
        model_path (str): Path to model.pt file
        
    Returns:
        ActorDepthCNN: Loaded actor network
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract actor_critic from checkpoint
    actor_critic_state = checkpoint['model_state_dict']
    
    # Create actor network with same parameters as training
    actor = ActorDepthCNN(
        num_obs_proprio=51,
        obs_depth_shape=(24, 32),
        num_actions=12,
        activation=get_activation("elu"),
        hidden_dims=[512, 256, 128]
    )
    
    # Load actor weights
    actor_state_dict = {}
    for key, value in actor_critic_state.items():
        if key.startswith('actor.'):
            # Remove 'actor.' prefix
            new_key = key[6:]  # Remove 'actor.' prefix
            actor_state_dict[new_key] = value
    
    actor.load_state_dict(actor_state_dict)
    actor.eval()
    
    return actor


def infer(observation_dict, actor_network):
    """
    Run inference on observation dict
    
    Args:
        observation_dict (dict): Dictionary containing 'proprio' and 'depth' keys
            - 'proprio': torch.Tensor of shape (51,) - proprioceptive observations
            - 'depth': torch.Tensor of shape (24, 32) - depth image
        actor_network (ActorDepthCNN): Loaded actor network
        
    Returns:
        torch.Tensor: Action tensor of shape (12,) - joint actions
    """
    with torch.no_grad():
        # Extract observations
        proprio = observation_dict['proprio']  # Shape: (51,)
        depth = observation_dict['depth']      # Shape: (24, 32)
        
        # Ensure tensors are float32 and have batch dimension
        if proprio.dim() == 1:
            proprio = proprio.unsqueeze(0)  # Shape: (1, 51)
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)      # Shape: (1, 24, 32)
            
        # Concatenate proprio and depth (flatten depth)
        depth_flat = depth.view(depth.shape[0], -1)  # Shape: (1, 768)
        combined_obs = torch.cat([proprio, depth_flat], dim=-1)  # Shape: (1, 819)
        
        # Run inference
        actions = actor_network(combined_obs)  # Shape: (1, 12)
        
        # Remove batch dimension
        return actions.squeeze(0)  # Shape: (12,)


def create_proprio_observation(
    base_lin_vel=None,
    base_ang_vel=None, 
    projected_gravity=None,
    base_rpy=None,
    velocity_commands=None,
    joint_pos=None,
    joint_vel=None,
    actions=None
):
    """
    Create proprioceptive observation tensor
    
    Args:
        base_lin_vel (torch.Tensor, optional): Base linear velocity (3,). Default: zeros
        base_ang_vel (torch.Tensor, optional): Base angular velocity (3,). Default: zeros
        projected_gravity (torch.Tensor, optional): Projected gravity vector (3,). Default: [0, 0, -1]
        base_rpy (torch.Tensor, optional): Base roll-pitch-yaw (3,). Default: zeros
        velocity_commands (torch.Tensor, optional): Velocity commands (3,). Default: zeros
        joint_pos (torch.Tensor, optional): Joint positions (12,). Default: zeros
        joint_vel (torch.Tensor, optional): Joint velocities (12,). Default: zeros
        actions (torch.Tensor, optional): Previous actions (12,). Default: zeros
        
    Returns:
        torch.Tensor: Concatenated proprioceptive observation (51,)
    """
    # Default values
    if base_lin_vel is None:
        base_lin_vel = torch.zeros(3)
    if base_ang_vel is None:
        base_ang_vel = torch.zeros(3)
    if projected_gravity is None:
        projected_gravity = torch.tensor([0.0, 0.0, -1.0])
    if base_rpy is None:
        base_rpy = torch.zeros(3)
    if velocity_commands is None:
        velocity_commands = torch.zeros(3)
    if joint_pos is None:
        joint_pos = torch.zeros(12)
    if joint_vel is None:
        joint_vel = torch.zeros(12)
    if actions is None:
        actions = torch.zeros(12)
    
    # Concatenate all proprioceptive data
    proprio = torch.cat([
        base_lin_vel,      # 3
        base_ang_vel,      # 3
        projected_gravity, # 3
        base_rpy,          # 3
        velocity_commands, # 3
        joint_pos,         # 12
        joint_vel,         # 12
        actions            # 12
    ])  # Total: 51
    
    return proprio


def create_example_observation():
    """
    Create example observation for testing
    
    Returns:
        dict: Example observation dictionary
    """
    # Create random example data
    proprio = torch.randn(51)  # Random proprioceptive data
    depth = torch.randn(24, 32)  # Random depth image
    
    return {
        'proprio': proprio,
        'depth': depth
    }


def create_realistic_observation():
    """
    Create more realistic observation for testing
    
    Returns:
        dict: Realistic observation dictionary
    """
    # Create realistic proprioceptive data
    proprio = create_proprio_observation(
        base_lin_vel=torch.tensor([0.5, 0.0, 0.0]),  # Moving forward
        base_ang_vel=torch.tensor([0.0, 0.0, 0.1]),  # Slight yaw rotation
        projected_gravity=torch.tensor([0.0, 0.0, -1.0]),  # Gravity pointing down
        base_rpy=torch.tensor([0.05, 0.02, 0.1]),  # Small roll/pitch/yaw
        velocity_commands=torch.tensor([0.5, 0.0, 0.0]),  # Forward command
        joint_pos=torch.tensor([0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5, 0.0, 0.0]),  # Typical joint positions
        joint_vel=torch.randn(12) * 0.1,  # Small joint velocities
        actions=torch.randn(12) * 0.1  # Previous actions
    )
    
    # Create depth image (closer objects = smaller values)
    depth = torch.rand(24, 32) * 2.0 + 0.3  # Range: 0.3 to 2.3 meters
    
    return {
        'proprio': proprio,
        'depth': depth
    }


if __name__ == "__main__":
    # Example usage
    print("Simple CNN Inference Example")
    print("=" * 40)
    
    # Test 1: Random observation
    print("Test 1: Random observation")
    obs_random = create_example_observation()
    print(f"Proprio shape: {obs_random['proprio'].shape}")
    print(f"Depth shape: {obs_random['depth'].shape}")
    
    # Test 2: Realistic observation
    print("\nTest 2: Realistic observation")
    obs_realistic = create_realistic_observation()
    print(f"Proprio shape: {obs_realistic['proprio'].shape}")
    print(f"Depth shape: {obs_realistic['depth'].shape}")
    print(f"Proprio values preview: {obs_realistic['proprio'][:10]}...")
    
    # Test 3: Custom proprio observation
    print("\nTest 3: Custom proprio observation")
    custom_proprio = create_proprio_observation(
        base_lin_vel=torch.tensor([1.0, 0.0, 0.0]),  # Fast forward
        velocity_commands=torch.tensor([1.0, 0.0, 0.0]),  # Forward command
        joint_pos=torch.ones(12) * 0.5  # All joints at 0.5 rad
    )
    print(f"Custom proprio shape: {custom_proprio.shape}")
    print(f"Custom proprio preview: {custom_proprio[:10]}...")
    
    # Create actor network (without loading weights for demo)
    actor = ActorDepthCNN(
        num_obs_proprio=51,
        obs_depth_shape=(24, 32),
        num_actions=12,
        activation=get_activation("elu"),
        hidden_dims=[256, 256, 128]
    )
    actor.eval()
    
    # Run inference on realistic observation
    action = infer(obs_realistic, actor)
    print(f"\nAction shape: {action.shape}")
    print(f"Action values: {action}")
    
    print("\n" + "=" * 50)
    print("Usage Examples:")
    print("=" * 50)
    print("1. Load model: actor = load_policy('path/to/model.pt')")
    print("2. Create observation:")
    print("   obs = {'proprio': create_proprio_observation(...), 'depth': depth_tensor}")
    print("3. Run inference: action = infer(obs, actor)")
    print("\nProprio observation components:")
    print("- base_lin_vel: Base linear velocity (3,)")
    print("- base_ang_vel: Base angular velocity (3,)")
    print("- projected_gravity: Gravity vector (3,)")
    print("- base_rpy: Roll-pitch-yaw (3,)")
    print("- velocity_commands: Velocity commands (3,)")
    print("- joint_pos: Joint positions (12,)")
    print("- joint_vel: Joint velocities (12,)")
    print("- actions: Previous actions (12,)")

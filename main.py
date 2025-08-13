import os
from src import generate_data_for_scenarios

def main():
    # Define the scenario types and config files
    config_files = ["no_discrimination.yaml", "single_discrimination.yaml", "multiple_discrimination.yaml",
                    "intersectional_discrimination.yaml", "compounded_discrimination.yaml"]
    
    base_output_folder = "generated_data"
    config_folder = "./config" 
    
    # seeds
    seeds = [42, 100, 111, 123, 789, 999, 1234, 2024, 3456, 5678]

    for seed in seeds:
        # Create seed-specific output folder
        seed_output_folder = os.path.join(base_output_folder, f"seed_{seed}")
        print(f"\n🎯 Generating data for seed {seed}...")
        # print(f"📁 Output folder: {seed_output_folder}")
        
        for config_file in config_files:
            # Create full path to config file
            config_path = os.path.join(config_folder, config_file)
            print(f"  🔧 Processing {config_file}...")
            generate_data_for_scenarios(config_path, [seed], seed_output_folder)
    
    print(f"\n✅ Data generation complete!")
    # print(f"📊 Generated data for {len(seeds)} seeds with {len(config_files)} scenarios each")
    
if __name__ == "__main__":
    main()
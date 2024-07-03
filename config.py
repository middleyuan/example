import subprocess
import os

def main():

    # configure the cmake build
    cmake_args = ['cmake',
                  '-DHYPERPARAM_GAMMA=0.99',
                  '-DHYPERPARAM_LAMBDA=0.95',
                  '-DHYPERPARAM_EPSILON_CLIP=0.2',
                  '-DHYPERPARAM_INITIAL_ACTION_STD=0.5',
                  '-DHYPERPARAM_ACTION_ENTROPY_COEFFICIENT=0.01',
                  '-DHYPERPARAM_POLICY_KL_EPSILON=1e-5',
                  '-DHYPERPARAM_N_WARMUP_STEPS_CRITIC=0',
                  '-DHYPERPARAM_N_WARMUP_STEPS_ACTOR=0',
                  '-DHYPERPARAM_ON_POLICY_RUNNER_STEPS_PER_ENV=1024',
                  '-DHYPERPARAM_BATCH_SIZE=256',
                  '-DHYPERPARAM_TOTAL_STEP_LIMIT=300000',
                  '-DHYPERPARAM_EPISODE_STEP_LIMIT=200',
                  '-DHYPERPARAM_ACTOR_HIDDEN_DIM=64',
                  '-DHYPERPARAM_ACTOR_NUM_LAYERS=3',
                  '-DHYPERPARAM_ACTOR_ACTIVATION_FUNCTION=1', # RELU 1,GELU 2,TANH 3,FAST_TANH 4,SIGMOID 5
                  '-DHYPERPARAM_CRITIC_HIDDEN_DIM=64',
                  '-DHYPERPARAM_CRITIC_NUM_LAYERS=3',
                  '-DHYPERPARAM_CRITIC_ACTIVATION_FUNCTION=1', # RELU 1,GELU 2,TANH 3,FAST_TANH 4,SIGMOID 5
                  '-DHYPERPARAM_N_EPOCHS=2',
                  '-DCMAKE_BUILD_TYPE=Release', '../..']
    if not os.path.exists('./build/test'):
        os.makedirs('./build/test')
    os.chdir('./build/test')
    cmake_process = subprocess.run(cmake_args, capture_output=True, text=True)
    
    if cmake_process.returncode != 0:
        print("CMake configuration failed:")
        print(cmake_process.stdout)
        print(cmake_process.stderr)
        return
    print(cmake_process.stdout.strip())

    # build the project
    build_args = ['cmake', '--build', '.']
    build_process = subprocess.run(build_args, capture_output=True, text=True)

    if build_process.returncode != 0:
        print("Build failed:")
        print(build_process.stdout)
        print(build_process.stderr)
        return
    
    # run the executable
    args = ['./my_pendulum']
    args += ['--seed=3', '--actor_alpha=0.01', '--critic_alpha=0.01']
    result = subprocess.run(args, capture_output=True, text=True)
    print(result.stdout.strip())
    # read the last line fo stdout
    final_reward = float(result.stdout.strip().split('\n')[-2].split(' ')[-1])
    print(f'Final reward: {final_reward}')


if __name__ == "__main__":
    main()
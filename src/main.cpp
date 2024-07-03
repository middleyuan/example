#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn_models/operations_cpu.h>

#include "../include/my_pendulum/my_pendulum.h"
#include "../include/my_pendulum/operations_generic.h"

#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>

#include <cxxopts.hpp>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using T = float;
using TI = typename DEVICE::index_t;
using PENDULUM_SPEC = MyPendulumSpecification<T, TI, MyPendulumParameters<T>>;
using ENVIRONMENT = MyPendulum<PENDULUM_SPEC>;
struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
    struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<T, TI>{
        static constexpr T GAMMA = static_cast<T>(HYPERPARAM_GAMMA);
        static constexpr T LAMBDA = static_cast<T>(HYPERPARAM_LAMBDA);
        static constexpr T EPSILON_CLIP = static_cast<T>(HYPERPARAM_EPSILON_CLIP);
        static constexpr T INITIAL_ACTION_STD = static_cast<T>(HYPERPARAM_INITIAL_ACTION_STD); // note this is NOT log(std) but actual std (log is applied at init)
        static constexpr bool LEARN_ACTION_STD = true;
        static constexpr T ACTION_ENTROPY_COEFFICIENT = static_cast<T>(HYPERPARAM_ACTION_ENTROPY_COEFFICIENT);
        static constexpr bool NORMALIZE_ADVANTAGE = true;
        static constexpr T POLICY_KL_EPSILON = static_cast<T>(HYPERPARAM_POLICY_KL_EPSILON);
        static constexpr TI N_WARMUP_STEPS_CRITIC = static_cast<TI>(HYPERPARAM_N_WARMUP_STEPS_CRITIC);
        static constexpr TI N_WARMUP_STEPS_ACTOR = static_cast<TI>(HYPERPARAM_N_WARMUP_STEPS_ACTOR);
        static constexpr TI N_EPOCHS = static_cast<TI>(HYPERPARAM_N_EPOCHS);;
    };

    static constexpr TI N_ENVIRONMENTS = 4;
    static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = static_cast<TI>(HYPERPARAM_ON_POLICY_RUNNER_STEPS_PER_ENV);
    static constexpr TI BATCH_SIZE = static_cast<TI>(HYPERPARAM_BATCH_SIZE);
    static constexpr TI TOTAL_STEP_LIMIT = static_cast<TI>(HYPERPARAM_TOTAL_STEP_LIMIT);
    static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT/(ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS) + 1;
    static constexpr TI EPISODE_STEP_LIMIT = static_cast<TI>(HYPERPARAM_EPISODE_STEP_LIMIT);

    static constexpr TI ACTOR_HIDDEN_DIM = static_cast<TI>(HYPERPARAM_ACTOR_HIDDEN_DIM);
    static constexpr TI ACTOR_NUM_LAYERS = static_cast<TI>(HYPERPARAM_ACTOR_NUM_LAYERS);
    static constexpr auto ACTOR_ACTIVATION_FUNCTION = static_cast<rlt::nn::activation_functions::ActivationFunction>(HYPERPARAM_ACTOR_ACTIVATION_FUNCTION);
    static constexpr TI CRITIC_HIDDEN_DIM = static_cast<TI>(HYPERPARAM_CRITIC_HIDDEN_DIM);
    static constexpr TI CRITIC_NUM_LAYERS = static_cast<TI>(HYPERPARAM_CRITIC_NUM_LAYERS);
    static constexpr auto CRITIC_ACTIVATION_FUNCTION = static_cast<rlt::nn::activation_functions::ActivationFunction>(HYPERPARAM_CRITIC_ACTIVATION_FUNCTION);

};
using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS>;
template <typename NEXT>
struct LOOP_EVAL_PARAMETERS: rlt::rl::loop::steps::evaluation::Parameters<T, TI, NEXT>{
    static constexpr TI EVALUATION_INTERVAL = NEXT::CORE_PARAMETERS::STEP_LIMIT;
    static constexpr TI NUM_EVALUATION_EPISODES = 10;
    static constexpr TI N_EVALUATIONS = NEXT::CORE_PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL + 1;
    static constexpr TI EPISODE_STEP_LIMIT = static_cast<TI>(HYPERPARAM_EPISODE_STEP_LIMIT);
};
#ifndef BENCHMARK
using LOOP_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CORE_CONFIG, LOOP_EVAL_PARAMETERS<LOOP_CORE_CONFIG>>;
#else
using LOOP_CONFIG = LOOP_CORE_CONFIG;
#endif
using LOOP_STATE = typename LOOP_CONFIG::template State<LOOP_CONFIG>;

// just for measuring the time
#include <chrono>
#include <iostream>

int main(int argc, char* argv[]){
    cxxopts::Options options("MyPendulum", "Pendulum RL Training");
    // specify the seed and hyperparameters
    options.add_options()
        ("seed", "Seed", cxxopts::value<int>())
        ("actor_alpha", "actor alpha", cxxopts::value<float>())
        ("critic_alpha", "critic alpha", cxxopts::value<float>());
    auto result = options.parse(argc, argv);

    DEVICE device;
    TI seed = result["seed"].as<int>();
    // TI seed = 1337;
    LOOP_STATE ls;
    rlt::malloc(device, ls);
    rlt::init(device, ls, seed);
    ls.actor_optimizer.parameters.alpha = result["actor_alpha"].as<float>();
    ls.critic_optimizer.parameters.alpha = result["critic_alpha"].as<float>();
    auto start_time = std::chrono::high_resolution_clock::now();
    while(!rlt::step(device, ls)){
        // do what ever you want here, e.g. poor man's learning rate scheduler:
        if(ls.step % 1 == 0){
            ls.actor_optimizer.parameters.alpha *= 0.9;
            ls.critic_optimizer.parameters.alpha *= 0.9;
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time-start_time;
    std::cout << "Training time: " << diff.count() << std::endl;
}
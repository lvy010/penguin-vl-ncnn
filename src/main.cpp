// SPDX-License-Identifier: Apache-2.0
//
// Penguin-VL-ncnn command line runner.
#include <cstdio>
#include <cstring>
#include <string>

#include "penguin_vl.h"

namespace {

void print_usage(const char* argv0) {
    std::printf(
        "Penguin-VL-ncnn runner\n"
        "Usage: %s --model <dir> [--image <path>] --prompt <text> [options]\n\n"
        "Options:\n"
        "  --model <dir>          model directory containing model.json (required)\n"
        "  --prompt <text>        user prompt (required)\n"
        "  --image <path>         image path for vision-language input\n"
        "  --threads <n>          CPU threads (default 4)\n"
        "  --max-new-tokens <n>   max tokens to generate (default 512)\n"
        "  --sample               enable sampling (default: greedy / deterministic)\n"
        "  --temperature <f>      sampling temperature (default 1.0)\n"
        "  --top-k <n>            top-k sampling (default 0 = off)\n"
        "  --top-p <f>            top-p sampling (default 1.0 = off)\n"
        "  --think                keep the assistant thinking turn open\n"
        "  -h, --help             show this help\n",
        argv0);
}

std::string next_arg(int& i, int argc, char** argv, const char* name) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "error: %s requires a value\n", name);
        std::exit(2);
    }
    return argv[++i];
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir, image_path, prompt;
    int threads = 4;
    pvl::GenerateConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model") model_dir = next_arg(i, argc, argv, "--model");
        else if (a == "--image") image_path = next_arg(i, argc, argv, "--image");
        else if (a == "--prompt") prompt = next_arg(i, argc, argv, "--prompt");
        else if (a == "--threads") threads = std::stoi(next_arg(i, argc, argv, "--threads"));
        else if (a == "--max-new-tokens") cfg.max_new_tokens = std::stoi(next_arg(i, argc, argv, "--max-new-tokens"));
        else if (a == "--sample") cfg.do_sample = true;
        else if (a == "--temperature") cfg.temperature = std::stof(next_arg(i, argc, argv, "--temperature"));
        else if (a == "--top-k") cfg.top_k = std::stoi(next_arg(i, argc, argv, "--top-k"));
        else if (a == "--top-p") cfg.top_p = std::stof(next_arg(i, argc, argv, "--top-p"));
        else if (a == "--think") cfg.add_think = true;
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); print_usage(argv[0]); return 2; }
    }

    if (model_dir.empty() || prompt.empty()) {
        print_usage(argv[0]);
        return 2;
    }

    try {
        pvl::PenguinVL model(model_dir, threads, /*use_vulkan=*/false);
        std::printf("Assistant: ");
        std::fflush(stdout);
        model.chat(prompt, image_path, cfg, [](const std::string& piece) {
            std::printf("%s", piece.c_str());
            std::fflush(stdout);
        });
        std::printf("\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nerror: %s\n", e.what());
        return 1;
    }
    return 0;
}

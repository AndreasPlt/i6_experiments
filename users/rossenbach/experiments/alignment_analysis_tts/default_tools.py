from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.tools.rasr import compile_rasr_binaries_i6mode


#RASR_BINARY_PATH = compile_rasr_binaries_i6mode(commit="907eec4f4e36c11153f6ab6b5dd7675116f909f6")
RASR_BINARY_PATH = compile_rasr_binaries_i6mode(branch="bene_unpushed_assert")
RASR_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_RASR_BINARY_PATH"


RETURNN_EXE = tk.Path("/u/rossenbach/bin/returnn/returnn_tf_dynamic_version_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
RETURNN_DATA_ROOT = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                          commit="37ba06ab2697e7af4de96037565fdf4f78acdb80").out_repository

RETURNN_RC_ROOT = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn", commit="f7c2c00e7a2b6c58c2bd5c1ef0fba711f9078de1").out_repository
RETURNN_RC_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_RC_ROOT"

RETURNN_COMMON = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn_common", commit="7e616c51ca278fce6e85b4f32ef396d782841c2f", checkout_folder_name="returnn_common").out_repository
RETURNN_COMMON.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_COMMON"

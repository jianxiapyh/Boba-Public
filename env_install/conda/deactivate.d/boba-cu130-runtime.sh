# Restore the caller's library search path exactly as it was before the Boba
# CUDA 13 environment was activated.
if [ "${_BOBA_CU130_RUNTIME_ACTIVE:-0}" = "1" ]; then
    if [ "${_BOBA_CU130_LD_LIBRARY_PATH_WAS_SET:-0}" = "1" ]; then
        export LD_LIBRARY_PATH="${_BOBA_CU130_SAVED_LD_LIBRARY_PATH-}"
    else
        unset LD_LIBRARY_PATH
    fi

    unset _BOBA_CU130_RUNTIME_ACTIVE
    unset _BOBA_CU130_LD_LIBRARY_PATH_WAS_SET
    unset _BOBA_CU130_SAVED_LD_LIBRARY_PATH
fi

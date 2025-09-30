# Compiling FFmpeg with NVIDIA GPU Acceleration and VMAF on Ubuntu

This guide provides a comprehensive walkthrough for compiling FFmpeg from source on an Ubuntu system equipped with an NVIDIA GPU. The resulting build will support NVIDIA's hardware encoding/decoding (NVENC/NVDEC), NPP filters (NVIDIA Performance Primitives), and CUDA-based VMAF (Video Multi-Method Assessment Fusion) for video quality assessment.

## Environment and Versions

Before you begin, ensure your system environment is similar to the configuration below. Version matching is crucial for a successful compilation.

- **GPU**: NVIDIA A6000
- **OS**: Ubuntu (or other Debian-based distributions)
- **NVIDIA Driver Version**: `535.161.08`
- **CUDA Version (from `nvidia-smi`)**: `12.2`
- **CUDA Toolkit Version**: `12.1` (This is the version used for compilation)
- **Target FFmpeg Version**: `6.1`
- **Target NVIDIA Codec Headers Version**: `sdk/12.1`

**Key Tip**: The version of the `NVIDIA Codec Headers` (`ffnvcodec`) must be compatible with the `CUDA Toolkit` version installed on your system and the version of `FFmpeg` you intend to compile.

## Compilation Steps

Please follow these steps in order.

### Step 1: Install System Dependencies

Open your terminal and install the essential tools and libraries required for compilation.

```bash
sudo apt-get update
sudo apt-get install -y build-essential yasm cmake libtool libc6 libc6-dev unzip wget libnuma1 libnuma-dev git

```

### Step 2: Install Meson and Ninja Build Tools

`libvmaf` uses `Meson` and `Ninja` for its build process. The default version of `Meson` in Ubuntu's repositories may be too old, so we'll install a newer version via `pip`.

1. **Install pip and the tools**
    
    ```bash
    sudo apt install python3-pip -y
    sudo pip3 install --upgrade meson ninja
    
    ```
    
2. **Verify the version and handle path issues**`libvmaf` requires Meson version ≥ 0.56.1.
    
    ```bash
    meson --version
    
    ```
    
    If the version is too low or the command is not found, it's likely because the newly installed executable is not in your system's `PATH`.
    
    ```bash
    # Find where meson was installed
    which meson
    # The output is usually /usr/local/bin/meson
    
    # Add its path to your environment (for the current session)
    export PATH="/usr/local/bin:$PATH"
    
    # Add it permanently to your ~/.bashrc
    echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    
    # (Optional) If an old version exists, you can update the symbolic link
    # sudo rm /usr/bin/meson
    # sudo ln -s /usr/local/bin/meson /usr/bin/meson
    
    ```
    

### Step 3: Install NVIDIA Codec Headers

These are the header files FFmpeg needs to interface with the NVIDIA driver for hardware-accelerated video processing.

1. **Clone the repository**
    
    ```bash
    git clone <https://github.com/FFmpeg/nv-codec-headers.git>
    cd nv-codec-headers
    
    ```
    
2. **Check out the branch that matches your CUDA Toolkit**
Since our CUDA Toolkit is version 12.1, we'll use the `sdk/12.1` branch.
    
    ```bash
    git checkout sdk/12.1
    
    ```
    
3. **Install the headers**
    
    ```bash
    sudo make install
    cd ..
    
    ```
    

### Step 4: Compile and Install VMAF (libvmaf)

To use the `libvmaf_cuda` filter in FFmpeg, we must first compile `libvmaf` from source with CUDA support enabled.

1. **Clone the VMAF repository**
    
    ```bash
    git clone <https://github.com/Netflix/vmaf.git>
    cd vmaf
    
    ```
    
2. * (Optional) Check out a stable release**
It's recommended to use a stable version, like `v2.3.1`.
    
    ```bash
    git checkout v2.3.1
    
    ```
    
3. **Configure the project with Meson**
Use the `meson setup` command to configure the build options, making sure to enable CUDA.
    
    **IMPORTANT**: If you have had a failed compilation attempt before, make sure to clean up the old build directory (`rm -rf libvmaf/build`) and any installed remnants (`sudo rm /usr/local/bin/vmaf`).
    
    Using the following command, which explicitly specifies CUDA paths, is highly recommended:
    
    ```bash
    meson setup libvmaf/build libvmaf \\
        --buildtype=release \\
        -Denable_cuda=true \\
        -Dcuda_include_dir=/usr/local/cuda/include \\
        -Dcuda_lib_dir=/usr/local/cuda/lib64 \\
        -Dcuda_compiler=/usr/local/cuda/bin/nvcc
    
    ```
    
4. **Compile and install**
    
    ```bash
    # Compile
    ninja -C libvmaf/build
    # Install
    sudo ninja -C libvmaf/build install
    
    ```
    
5. **Refresh the dynamic library cache**
This crucial step ensures the system can find the newly installed `libvmaf.so`.
    
    ```bash
    sudo ldconfig
    cd ..
    
    ```
    

### Step 5: Compile and Install FFmpeg

With all dependencies in place, we can now compile FFmpeg itself.

1. **Clone the FFmpeg repository**
    
    ```bash
    git clone <https://github.com/FFmpeg/FFmpeg.git>
    cd FFmpeg
    
    ```
    
2. **Check out the target version**
    
    ```bash
    git checkout release/6.1
    
    ```
    
3. **Configure the build**
This command configures FFmpeg to enable NVIDIA features and `libvmaf`.
    
    ```bash
    ./configure \\
      --enable-nonfree \\
      --enable-cuda-nvcc \\
      --enable-libnpp \\
      --extra-cflags=-I/usr/local/cuda/include \\
      --extra-ldflags=-L/usr/local/cuda/lib64 \\
      --disable-static \\
      --enable-shared \\
      --enable-libvmaf
    
    ```
    
4. **Compile and install**
Use the `j` flag to parallelize the compilation and speed it up (e.g., using `8` cores).
    
    ```bash
    make -j8
    sudo make install
    
    ```
    
5. **Refresh the dynamic library cache again**
This allows the system to recognize the newly installed FFmpeg libraries.
    
    ```bash
    sudo ldconfig
    
    ```
    

## Verify the Installation

After installation, you can run these commands to confirm that FFmpeg was built correctly.

1. **Check version and configuration flags**
    
    ```bash
    ffmpeg -version
    
    ```
    
    The output should include the flags you enabled, such as `--enable-cuda-nvcc`, `--enable-libnpp`, and `--enable-libvmaf`.
    
2. **Check for available hardware accelerators**
    
    ```bash
    ffmpeg -hwaccels
    
    ```
    
    The list should include `cuda`.
    
3. **Check for NVIDIA encoders**
    
    ```bash
    ffmpeg -encoders | grep nvenc
    
    ```
    
    You should see encoders like `h264_nvenc`, `hevc_nvenc`, etc.
    

## Usage Example

This command demonstrates how to use hardware acceleration (`scale_npp`) and the CUDA-based VMAF filter to assess video quality.

```bash
ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i distorted_video.mp4 -i reference_video.mp4 \\
-filter_complex \\
"[0:v]scale_npp=format=yuv420p[dis];[1:v]scale_npp=format=yuv420p[ref];[dis][ref]libvmaf_cuda" \\
-f null -

```

- `hwaccel cuda`: Enables CUDA hardware decoding.
- `scale_npp`: Uses the NVIDIA NPP library for GPU-accelerated video scaling.
- `libvmaf_cuda`: Uses the CUDA-based VMAF filter for calculations.

## Troubleshooting

### Issue 1: VMAF compilation fails with `vcs_version.h: No such file or directory`

- **Cause**: This error typically occurs if you downloaded the VMAF source code as a ZIP archive instead of using `git clone`. The build script relies on the `.git` directory to generate version header files.
- **Solution**:
    1. **Recommended**: Always use `git clone` to get the source code.
        
        ```bash
        git clone <https://github.com/Netflix/vmaf.git>
        
        ```
        
    2. **Alternative**: If you cannot use git, you can create the version file manually.
        
        ```bash
        # Assuming you are using version v2.3.1
        echo '#define VMAF_VCS_VERSION "v2.3.1"' > libvmaf/src/vcs_version.h
        
        ```
        

### Issue 2: FFmpeg `configure` fails with error about Video Codec SDK version being too low

- **Error Message**: Something like `ERROR: nvenc requested, but NVIDIA Video Codec SDK 12.0 or later is required.`
- **Cause**: This means the version of `nv-codec-headers` you checked out is not compatible with your NVIDIA driver, CUDA Toolkit, or the version of FFmpeg you are building. For instance, FFmpeg 6.1 may require newer headers than you have, or your headers might be too new for your driver/SDK.
- **Solution**:
    1. Carefully re-check your [NVIDIA Driver](https://www.notion.so/gemini-update-27de345300e4809799a1c1295bea1d69?pvs=21) and [CUDA Toolkit](https://www.notion.so/gemini-update-27de345300e4809799a1c1295bea1d69?pvs=21) versions.
    2. Go back to [Step 3: Install NVIDIA Codec Headers](https://www.notion.so/gemini-update-27de345300e4809799a1c1295bea1d69?pvs=21) and ensure you `git checkout` the branch that best matches your environment (e.g., `sdk/12.1`).
    3. Consult the [Official NVIDIA FFmpeg Guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/11.1/ffmpeg-with-nvidia-gpu/index.html) or the `nv-codec-headers` repository to confirm version compatibility.

## References

- [VMAF](https://github.com/Netflix/vmaf)
- [FFmpeg Official Source](https://github.com/FFmpeg/FFmpeg/tree/release/6.1)
- [NVIDIA Codec Headers Source](https://github.com/FFmpeg/nv-codec-headers/tree/sdk/12.1)
- [Official NVIDIA Guide for Compiling FFmpeg](https://docs.nvidia.com/video-technologies/video-codec-sdk/11.1/ffmpeg-with-nvidia-gpu/index.html)
- [Related CSDN Blog Post](https://blog.csdn.net/qq_43513908/article/details/138161139)

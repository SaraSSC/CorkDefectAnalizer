 @echo off
REM Use either wget or curl to download the checkpoints
where wget >nul 2>&1
if %errorlevel%==0 (
    set CMD=wget
) else (
    where curl >nul 2>&1
    if %errorlevel%==0 (
        set CMD=curl -L -O
    ) else (
        echo Please install wget or curl to download the checkpoints.
        exit /b 1
    )
)

REM Define the URLs for SAM 2 checkpoints
REM set SAM2_BASE_URL=https://dl.fbaipublicfiles.com/segment_anything_2/072824
REM set sam2_hiera_t_url=%SAM2_BASE_URL%/sam2_hiera_tiny.pt
REM set sam2_hiera_s_url=%SAM2_BASE_URL%/sam2_hiera_small.pt
REM set sam2_hiera_b_plus_url=%SAM2_BASE_URL%/sam2_hiera_base_plus.pt
REM set sam2_hiera_l_url=%SAM2_BASE_URL%/sam2_hiera_large.pt

REM Download each of the four checkpoints
REM echo Downloading sam2_hiera_tiny.pt checkpoint...
REM %CMD% %sam2_hiera_t_url% || (echo Failed to download checkpoint from %sam2_hiera_t_url% & exit /b 1)

REM echo Downloading sam2_hiera_small.pt checkpoint...
REM %CMD% %sam2_hiera_s_url% || (echo Failed to download checkpoint from %sam2_hiera_s_url% & exit /b 1)

REM echo Downloading sam2_hiera_base_plus.pt checkpoint...
REM %CMD% %sam2_hiera_b_plus_url% || (echo Failed to download checkpoint from %sam2_hiera_b_plus_url% & exit /b 1)

REM echo Downloading sam2_hiera_large.pt checkpoint...
REM %CMD% %sam2_hiera_l_url% || (echo Failed to download checkpoint from %sam2_hiera_l_url% & exit /b 1)

REM Define the URLs for SAM 2.1 checkpoints
set SAM2p1_BASE_URL=https://dl.fbaipublicfiles.com/segment_anything_2/092824
set sam2p1_hiera_t_url=%SAM2p1_BASE_URL%/sam2.1_hiera_tiny.pt
set sam2p1_hiera_s_url=%SAM2p1_BASE_URL%/sam2.1_hiera_small.pt
set sam2p1_hiera_b_plus_url=%SAM2p1_BASE_URL%/sam2.1_hiera_base_plus.pt
set sam2p1_hiera_l_url=%SAM2p1_BASE_URL%/sam2.1_hiera_large.pt

REM SAM 2.1 checkpoints
echo Downloading sam2.1_hiera_tiny.pt checkpoint...
%CMD% %sam2p1_hiera_t_url% || (echo Failed to download checkpoint from %sam2p1_hiera_t_url% & exit /b 1)

echo Downloading sam2.1_hiera_small.pt checkpoint...
%CMD% %sam2p1_hiera_s_url% || (echo Failed to download checkpoint from %sam2p1_hiera_s_url% & exit /b 1)

echo Downloading sam2.1_hiera_base_plus.pt checkpoint...
%CMD% %sam2p1_hiera_b_plus_url% || (echo Failed to download checkpoint from %sam2p1_hiera_b_plus_url% & exit /b 1)

echo Downloading sam2.1_hiera_large.pt checkpoint...
%CMD% %sam2p1_hiera_l_url% || (echo Failed to download checkpoint from %sam2p1_hiera_l_url% & exit /b 1)

echo All checkpoints are downloaded successfully.
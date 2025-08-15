#!/bin/bash
# Install CUDA on manylinux docker image.
set -eux

VER=${1:-12.8}
VER=${VER//./-}  # Convert version to format used in package names

dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

# Install GCC 13

dnf -y install gcc-toolset-13
dnf -y remove gcc-toolset-14-*
echo ". /opt/rh/gcc-toolset-13/enable" > /etc/profile.d/gcc.sh
chmod +x /etc/profile.d/gcc.sh

# Create a fake package to stop cuda from stupidly installing gcc-8.5

dnf -y install rpm-build

mkdir -p ~/rpmbuild/{SPECS,RPMS,SOURCES}
cd ~/rpmbuild
cat > SPECS/gcc-dummy.spec <<EOF
Name:           gcc-dummy
Version:        13
Release:        1%{?dist}
Summary:        Dummy package to provide gcc-c++
License:        MIT
BuildArch:      noarch
Provides:       gcc-c++ = 13

%description
Dummy package that provides gcc-c++ capabilities without actual compiler

%files

%changelog
* Wed Feb 12 2025 User <user@example.com> - 8.5.0-1
- Initial package
EOF
rpmbuild -bb SPECS/gcc-dummy.spec
rpm -ivh ~/rpmbuild/RPMS/noarch/gcc-dummy*.rpm --nodeps

# Install CUDA

dnf -y install \
    cuda-compiler-"${VER}" \
    cuda-minimal-build-"${VER}" \
    cuda-nvtx-"${VER}" \
    cuda-nvrtc-devel-"${VER}"

    # cuda-libraries-devel-${VER} \
# dnf clean all


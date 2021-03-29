include(${SPHERAL_ROOT_DIR}/cmake/tpl/util/Install_Python_distutils_library.cmake)

set(ATS_DIST ats-5.2.tar.gz)
set(ATS_URL "https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/ats/${ATS_DIST}")
set(ATS_CACHE ${CACHE_DIR}/${ATS_DIST})

Install_Python_distutils_library(ats ${ATS_CACHE} ${ATS_URL} "ats-5.2")

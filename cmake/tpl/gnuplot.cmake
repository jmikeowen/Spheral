include(${SPHERAL_ROOT_DIR}/cmake/tpl/Install_Python_distutils_library.cmake)

set(GNUPLOT_DIST gnuplot-py-1.8.tar.gz)
set(GNUPLOT_URL http://downloads.sourceforge.net/gnuplot-py/${GNUPLOT_DIST})
set(GNUPLOT_CACHE ${CACHE_DIR}/${GNUPLOT_DIST})

Install_Python_distutils_library(gnuplot ${GNUPLOT_CACHE} ${GNUPLOT_URL} "gnuplot-py-1.8")

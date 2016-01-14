#include "g2o/core/factory.h"
#include "edge_se3_linear_acceleration.h"



//ds register custom g2o types
namespace g2o
{
    G2O_REGISTER_TYPE( EDGE_SE3_LINEAR_ACCELERATION, EdgeSE3LinearAcceleration )

    //ds if drawing is enabled
    #ifdef G2O_HAVE_OPENGL

        G2O_REGISTER_ACTION( EdgeSE3LinearAccelerationDrawAction )

    #endif
}

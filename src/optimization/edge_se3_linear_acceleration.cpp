#include "edge_se3_linear_acceleration.h"

#include "g2o/types/slam3d/isometry3d_gradients.h"
#include <iostream>

#ifdef G2O_HAVE_OPENGL
#include "g2o/stuff/opengl_wrapper.h"
#include "g2o/stuff/opengl_primitives.h"
#endif

namespace g2o {
  using namespace std;

  // point to camera projection, monocular
  EdgeSE3LinearAcceleration::EdgeSE3LinearAcceleration() : BaseUnaryEdge<3, Vector3D, VertexSE3>() {
    setMeasurement(Vector3D(0,0,-1));
    information().setIdentity();
    _cache = 0;
    _offsetParam = 0;
    resizeParameters(1);
    installParameter(_offsetParam, 0);
  }


  bool EdgeSE3LinearAcceleration::resolveCaches(){
    assert(_offsetParam);
    ParameterVector pv(1);
    pv[0]=_offsetParam;
    resolveCache(_cache, (OptimizableGraph::Vertex*)_vertices[0],"CACHE_SE3_OFFSET",pv);
    return _cache != 0;
  }



    bool EdgeSE3LinearAcceleration::read( std::istream& is )
    {
        //ds retrieve parameter id
        int pid;
        is >> pid;
        if( !setParameterId( 0, pid ) )
        {
            return false;
        }

        //measured keypoint
        Vector3D vecMeasurement;

        for( int i = 0; i < 3; i++ )
        {
            is >> vecMeasurement[i];
        }

        setMeasurement( vecMeasurement );

        // don't need this if we don't use it in error calculation (???)
        // information matrix is the identity for features, could be changed to allow arbitrary covariances
        if( is.bad( ) )
        {
            return false;
        }
        for( int i = 0; i < information( ).rows( ) && is.good( ); i++ )
        {
            for( int j = i; j < information( ).cols( ) && is.good( ); j++ )
            {
                is >> information( )(i,j);
                if( i != j )
                {
                    information( )(j,i) = information( )(i,j);
                }
            }
        }
        if( is.bad( ) )
        {
            //  we overwrite the information matrix
            information( ).setIdentity( );
        }

        //ds success if still here
        return true;
    }

    bool EdgeSE3LinearAcceleration::write( std::ostream& os ) const
    {
        //ds log id
        os << _offsetParam->id( ) <<  " ";

        //ds write measurement
        for( int i=0; i<3; i++ )
        {
            os  << _measurement[i] << " ";
        }

        //ds information matrix (3x3)
        for( int i=0; i<information( ).rows( ); i++ )
        {
            for( int j=i; j<information( ).cols( ); j++ )
            {
                os <<  information( )(i,j) << " ";
            }
        }

        return os.good( );
    }


    void EdgeSE3LinearAcceleration::computeError( )
    {
        //ds get the measurement into world frame
        const Vector3D vecMeasurementWORLD( _cache->n2w( ).linear( )*_measurement );

        //ds and compare against normed gravity (- -1.0 -> +1.0)
        _error = vecMeasurementWORLD-Vector3D( 0.0, 0.0, -1.0 );

        //std::printf( "measurement: (%f, %f, %f)\n", vecMeasurementWORLD(0), vecMeasurementWORLD(1), vecMeasurementWORLD(2) );
        //std::printf( "error: (%f, %f, %f)\n", _error(0), _error(1), _error(2) );
    }

#ifdef G2O_HAVE_OPENGL

    EdgeSE3LinearAccelerationDrawAction::EdgeSE3LinearAccelerationDrawAction( ): DrawAction( typeid( EdgeSE3LinearAcceleration ).name( ) ){ }

    HyperGraphElementAction* EdgeSE3LinearAccelerationDrawAction::operator( )( HyperGraph::HyperGraphElement* element, HyperGraphElementAction::Parameters*  params_ )
    {
        if( typeid( *element ).name( ) != _typeName )
        {
            return 0;
        }

        refreshPropertyPtrs( params_ );

        if( ! _previousParams )
        {
            return this;
        }

        if( _show && !_show->value( ) )
        {
            return this;
        }

        const EdgeSE3LinearAcceleration* cMeasurement = static_cast< EdgeSE3LinearAcceleration* >( element );
        const ParameterSE3Offset* cParameterOffset    = static_cast< const ParameterSE3Offset* >( cMeasurement->parameter( 0 ) );
        const VertexSE3* cPosition                    = static_cast< VertexSE3* >( cMeasurement->vertices( )[0] );

        if( !cPosition )
        {
            return this;
        }

        //ds get measurement into world
        const Isometry3D matTransformationLEFTtoWORLD = cPosition->estimate( );
        const Isometry3D matTransformationIMUtoWORLD  = matTransformationLEFTtoWORLD*cParameterOffset->offset( );
        const Vector3D vecPositionWORLD( matTransformationIMUtoWORLD.translation( ) );
        const Vector3D vecMeasurementWORLD( matTransformationIMUtoWORLD.linear( )*cMeasurement->measurement( ) );

        glColor3f( LANDMARK_EDGE_COLOR );
        glPushAttrib( GL_ENABLE_BIT );
        glDisable( GL_LIGHTING );
        glBegin( GL_LINES );
        glVertex3f( static_cast< float >( vecPositionWORLD(0) ),static_cast< float >( vecPositionWORLD(1) ),static_cast< float >( vecPositionWORLD(2) ) );
        glVertex3f( static_cast< float >( vecPositionWORLD(0) + vecMeasurementWORLD(0) ),
                    static_cast< float >( vecPositionWORLD(1) + vecMeasurementWORLD(1) ),
                    static_cast< float >( vecPositionWORLD(2) + vecMeasurementWORLD(2) ) );
        glEnd( );
        glPopAttrib( );
        return this;
    }

#endif

}

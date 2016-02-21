#ifndef CWRAPPEROPENCV_H_
#define CWRAPPEROPENCV_H_

#include <Eigen/Core>
#include <opencv/cv.h>

class CWrapperOpenCV
{

//ds methods
public:

    template< typename tType, uint32_t uRows, uint32_t uCols > inline static const Eigen::Matrix< tType, uRows, uCols > fromCVMatrix( const cv::Mat& p_matCV )
    {
        //ds allocate eigen matrix
        Eigen::Matrix< tType, uRows, uCols > matEigen;

        //ds fill the matrix (column major)
        for( uint32_t u = 0; u < uRows; ++u )
        {
            for( uint32_t v = 0; v < uCols; ++v )
            {
                matEigen( u, v ) = p_matCV.at< tType >( u, v );
            }
        }

        return matEigen;
    }

    template< typename tType, uint32_t uRows > inline static const Eigen::Matrix< tType, uRows, 1 > fromCVVector( const cv::Vec< tType, uRows >& p_vecCV )
    {
        //ds allocate eigen matrix
        Eigen::Matrix< tType, uRows, 1 > vecEigen;

        //ds fill the vector (column major)
        for( uint32_t u = 0; u < uRows; ++u )
        {
            vecEigen( u ) = p_vecCV( u );
        }

        return vecEigen;
    }

    template< typename tType, uint32_t uRows > inline static const cv::Vec< tType, uRows > toCVVector( const Eigen::Matrix< tType, uRows, 1 >& p_vecEigen )
    {
        //ds allocate cv vector
        cv::Vec< tType, uRows > vecCV;

        //ds fill the vector (column major)
        for( uint32_t u = 0; u < uRows; ++u )
        {
            vecCV( u ) = p_vecEigen( u );
        }

        return vecCV;
    }

    //ds overloads
    inline static const cv::Vec3d toCVVector( const Eigen::Vector3d& p_vecEigen )
    {
        //ds allocate cv vector
        cv::Vec3d vecCV;

        //ds fill the vector (column major)
        for( uint32_t u = 0; u < 3; ++u )
        {
            vecCV( u ) = p_vecEigen( u );
        }

        return vecCV;
    }
    inline static const cv::Vec4d toCVVector( const Eigen::Vector4d& p_vecEigen )
    {
        //ds allocate cv vector
        cv::Vec4d vecCV;

        //ds fill the vector (column major)
        for( uint32_t u = 0; u < 4; ++u )
        {
            vecCV( u ) = p_vecEigen( u );
        }

        return vecCV;
    }
    inline static const cv::Point2d toCVVector( const Eigen::Vector2d& p_vecEigen )
    {
        //ds allocate cv vector
        return cv::Vec2d( p_vecEigen(0), p_vecEigen(1) );
    }
    inline static const cv::Mat_< double > toCVMatrix( const Eigen::Matrix< double, 3, 3 >& p_matEigen )
    {
        //ds allocate cv vector
        cv::Mat_< double > matCV( 3, 3 );

        //ds fill the vector (column major)
        for( uint32_t u = 0; u < 3; ++u )
        {
            for( uint32_t v = 0; v < 3; ++v )
            {
                matCV.at< double >( u, v ) = p_matEigen( u, v );
            }
        }

        return matCV;
    }
    inline static const cv::Mat_< double > toCVMatrix( const Eigen::Matrix< double, 3, 4 >& p_matEigen )
    {
        //ds allocate cv vector
        cv::Mat_< double > matCV( 3, 4 );

        //ds fill the vector (column major)
        for( uint32_t u = 0; u < 3; ++u )
        {
            for( uint32_t v = 0; v < 4; ++v )
            {
                matCV.at< double >( u, v ) = p_matEigen( u, v );
            }
        }

        return matCV;
    }
    inline static const Eigen::Vector2d fromCVVector( const cv::Vec2d& p_vecCV )
    {
        return Eigen::Vector2d( p_vecCV(0), p_vecCV(1) );
    }
    inline static const Eigen::Vector2d fromCVVector( const cv::Point2d& p_vecCV )
    {
        return Eigen::Vector2d( p_vecCV.x, p_vecCV.y );
    }

    inline static const Eigen::Vector2f fromCV( const cv::Point2d& p_vecCV )
    {
        return Eigen::Vector2f( p_vecCV.x, p_vecCV.y );
    }

    inline static const Eigen::Vector2d getInterDistance( const Eigen::Vector2d& p_vecA, const cv::Point2d& p_ptB )
    {
        return Eigen::Vector2d( p_vecA.x( )-p_ptB.x, p_vecA.y( )-p_ptB.y );
    }
    inline static const Eigen::Vector2d getInterDistance( const cv::Point2d& p_ptA, const Eigen::Vector2d& p_vecB )
    {
        return Eigen::Vector2d( p_ptA.x-p_vecB.x( ), p_ptA.y-p_vecB.y( ) );
    }

    //ds converts descriptors from cv::Mat to the current descriptor vector format
    inline static const Eigen::Matrix< double, 256, 1 > getDescriptorVector( const cv::Mat& p_cDescriptor )
    {
        //ds return vector
        Eigen::Matrix< double, 256, 1 > vecDescriptor;

        //ds compute bytes (as  opencv descriptors are bytewise)
        const uint32_t uDescriptorSizeBytes = 32;

        //ds loop over all bytes
        for( uint32_t u = 0; u < uDescriptorSizeBytes; ++u )
        {
            //ds get minimal datafrom cv::mat
            const uchar chValue = p_cDescriptor.at< uchar >( u );

            //ds get bitstring
            for( uint8_t v = 0; v < 8; ++v )
            {
                vecDescriptor[u*8+v] = ( chValue >> v ) & 1;
            }
        }

        return vecDescriptor;
    }

};

#endif //#define CWRAPPEROPENCV_H_

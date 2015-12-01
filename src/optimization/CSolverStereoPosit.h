#ifndef CSOLVERSTEREOPOSIT_H
#define CSOLVERSTEREOPOSIT_H

#include "types/CLandmark.h"



class CSolverStereoPosit
{

//ds eigen memory alignment
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

//ds own structures
public:

    struct CMatch
    {
        CLandmark* pLandmark;
        const CPoint3DWORLD vecPointXYZWORLD;
        const CPoint3DCAMERA vecPointXYZLEFT;
        const cv::Point2f ptUVLEFT;
        const cv::Point2f ptUVRIGHT;
        const CDescriptor matDescriptorLEFT;
        const CDescriptor matDescriptorRIGHT;

        CMatch( CLandmark* p_pLandmark,
                                      const CPoint3DCAMERA& p_vecPointXYZWORLD,
                                      const CPoint3DCAMERA& p_vecPointXYZLEFT,
                                      const cv::Point2f& p_ptUVLEFT,
                                      const cv::Point2f& p_ptUVRIGHT,
                                      const CDescriptor& p_matDescriptorLEFT,
                                      const CDescriptor& p_matDescriptorRIGHT ): pLandmark( p_pLandmark ),
                                                                                 vecPointXYZWORLD( p_vecPointXYZWORLD ),
                                                                                 vecPointXYZLEFT( p_vecPointXYZLEFT ),
                                                                                 ptUVLEFT( p_ptUVLEFT ),
                                                                                 ptUVRIGHT( p_ptUVRIGHT ),
                                                                                 matDescriptorLEFT( p_matDescriptorLEFT ),
                                                                                 matDescriptorRIGHT( p_matDescriptorRIGHT )
        {
            //ds nothing to do
        }
        ~CMatch( )
        {
            //ds nothing to do
        }
    };



//ds allocation
public:

    CSolverStereoPosit( const MatrixProjection& p_matProjectionLEFT,
                        const MatrixProjection& p_matProjectionRIGHT ): m_matProjectionLEFT( p_matProjectionLEFT ),
                                                                        m_matProjectionRIGHT( p_matProjectionRIGHT )
    {
        //ds nothing to do
    }



//ds access
public:

    const Eigen::Isometry3d getTransformationWORLDtoLEFT( const Eigen::Isometry3d& p_matTransformationWORLDtoLEFTLAST,
                                                          const Eigen::Vector3d& p_vecTranslationIMU,
                                                          const Eigen::Isometry3d& p_matTransformationWORLDtoLEFTESTIMATE,
                                                          const std::vector< CSolverStereoPosit::CMatch >& p_vecMeasurements );

    const double getDurationTotalSeconds( ) const { return m_dDurationTotalSeconds; }



//ds internals
private:

    //ds dynamic internals
    Eigen::Isometry3d m_matTransformationWORLDtoLEFT;
    Eigen::Matrix< double, 6, 6 > m_matH;
    Eigen::Matrix< double, 6, 1 > m_vecB;
    double m_dErrorTotalPreviousSquaredPixels = 0.0;

    //ds constant internals
    const MatrixProjection m_matProjectionLEFT;
    const MatrixProjection m_matProjectionRIGHT;
    const uint32_t m_uMinimumPointsForPoseOptimization = 25;
    const uint32_t m_uMinimumInliersPoseOptimization   = 15;
    const uint32_t m_uCapIterationsPoseOptimization    = 100;
    const double m_dMaximumErrorInlierPixelsL2         = 10.0;
    const double m_dMaximumRISK                        = 2.0; //2.0
    const double m_dConvergenceDelta                   = 1e-5;

    //ds precision settings
    const double m_dMinimumTranslationMetersL2 = 0.001;
    const double m_dMinimumRotationRadL2       = 0.0001;

    //ds timing
    double m_dDurationTotalSeconds = 0.0;

};

#endif //CSOLVERSTEREOPOSIT

#include "CViewerBTree.h"

#include <QKeyEvent>
#include <QGLViewer/manipulatedFrame.h>



CViewerBTree::CViewerBTree( const std::shared_ptr< CBITree< MAXIMUM_DISTANCE_HAMMING, BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS > > p_pRoot ): m_pBTree( p_pRoot ), m_pQuadratic( gluNewQuadric( ) )
{
    m_vecNodesQUERY.clear( );
}
CViewerBTree::~CViewerBTree( )
{
    //ds nothing to do
}

void CViewerBTree::draw( )
{
    glPushAttrib( GL_ENABLE_BIT );
    glDisable( GL_LIGHTING );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glEnable( GL_BLEND );
    glPointSize( 5.0 );
    glLineWidth( 1.0 );

    //ds draw WORLD coordinate frame
    glColor3f( 0.0, 0.0, 0.0 );
    drawAxis( 1.0 );

    //ds check if root is available
    if( 0 != m_pBTree->getRoot( ) )
    {
        //ds root node
        const Eigen::Vector3d vecRootPosition( 0.0, 0.0, 0.0 );
        const uint32_t uRootAngleDegrees( 0 );


        /*ds draw a  connecting line
        glColor3f( 0.0, 0.0, 1.0 );
        glBegin( GL_LINES );
        glVertex3f( 0.0, 0.0, 0.0 );
        glVertex3f( 0.1, 0.1, 0.1 );
        glEnd( );*/

        glBegin( GL_POINTS );

        //ds recursive function plotting all tree points in their correct locations
        _plotTreeNodesRecursive( m_pBTree->getRoot( ), vecRootPosition, uRootAngleDegrees );

        //ds plot query points in red
        for( const std::pair< Eigen::Vector3d, bool >& prNode: m_vecNodesQUERY )
        {
            //ds if matched
            if( prNode.second )
            {
                //ds blue
                glColor3f( 0.0, 0.0, 1.0 );
            }
            else
            {
                //ds red
                glColor3f( 1.0, 0.0, 0.0 );
            }

            //ds push the point
            glVertex3d( prNode.first.x( ), prNode.first.y( ), prNode.first.z( ) );
        }

        //ds all points drawn
        glEnd( );
    }

    glPopAttrib( ); //GL_ENABLE_BIT
}

void CViewerBTree::init( )
{
    //ds initialize
    setSceneRadius( 25.0 );
    setBackgroundColor( QColor( 255, 255, 255 ) );
}

QString CViewerBTree::helpString( ) const
{
  QString text("<h2>HUBBA BABA</h2>");
  return text;
}

void CViewerBTree::manualDraw( )
{
    draw( );
    updateGL( );
}

void CViewerBTree::highlightQUERY( const std::shared_ptr< const std::vector< CDescriptorBRIEF< DESCRIPTOR_SIZE_BITS > > > p_vecDescriptorsQUERY, const UIDKeyFrame& p_IDQUERY )
{
    //ds new query vector
    m_vecNodesQUERY.clear( );

    //ds for each descriptor
    for( const CDescriptorBRIEF< DESCRIPTOR_SIZE_BITS >& cDescriptorQUERY: *p_vecDescriptorsQUERY )
    {
        //ds also mimic location behavior
        Eigen::Vector3d vecPosition( 0.0, 0.0, 0.0 );
        uint32_t uAngleDegrees = 0;

        //ds traverse tree to find this descriptor
        const CBNode< BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS >* pNodeCurrent = m_pBTree->getRoot( );
        while( pNodeCurrent )
        {
            //ds spacing factor
            const double dSpacing = 0.1*pNodeCurrent->uDepth;

            //ds z is directly proportional to the depth value
            vecPosition.z( ) = 4*dSpacing;

            //ds if this node has leaves (is splittable)
            if( pNodeCurrent->bHasLeaves )
            {
                //ds check the split bit and go deeper
                if( cDescriptorQUERY.vecData[pNodeCurrent->uIndexSplitBit] )
                {
                    pNodeCurrent = pNodeCurrent->pLeafOnes;

                    //ds check if we have to fork in x or y
                    if( 0 == uAngleDegrees%2 )
                    {
                        //ds split in x
                        vecPosition.y( ) += dSpacing;
                    }
                    else
                    {
                        //ds same y coordinate for the split
                        vecPosition.x( ) += dSpacing;
                    }
                }
                else
                {
                    pNodeCurrent = pNodeCurrent->pLeafZeros;

                    //ds check if we have to fork in x or y
                    if( 0 == uAngleDegrees%2 )
                    {
                        //ds same y coordinate for the split
                        vecPosition.y( ) -= dSpacing;
                    }
                    else
                    {
                        //ds same y coordinate for the split
                        vecPosition.x( ) -= dSpacing;
                    }
                }

                ++uAngleDegrees;
            }
            else
            {
                //ds match success
                bool bMatchFound = false;

                //ds check current descriptors in this node and exit
                for( const CDescriptorBRIEF< DESCRIPTOR_SIZE_BITS >& cDescriptorTRAIN: pNodeCurrent->vecDescriptors )
                {
                    if( MAXIMUM_DISTANCE_HAMMING > CBNode< BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS >::getDistanceHamming( cDescriptorQUERY.vecData, cDescriptorTRAIN.vecData ) )
                    {
                        //ds also check for query id
                        if( p_IDQUERY == cDescriptorTRAIN.uIDKeyFrame )
                        {
                            bMatchFound = true;
                            break;
                        }
                    }
                }

                //ds save the node
                m_vecNodesQUERY.push_back( std::make_pair( vecPosition, bMatchFound ) );
                break;
            }
        }
    }
}

/*void CViewerScene::keyPressEvent( QKeyEvent* p_pEvent )
{
    switch( p_pEvent->key( ) )
    {
        default:
        {
            QGLViewer::close( );
            break;
        }
    }
}*/

void CViewerBTree::_drawBox(GLfloat l, GLfloat w, GLfloat h) const
{
    GLfloat sx = l*0.5f;
    GLfloat sy = w*0.5f;
    GLfloat sz = h*0.5f;

    glBegin(GL_QUADS);
    // bottom
    glNormal3f( 0.0f, 0.0f,-1.0f);
    glVertex3f(-sx, -sy, -sz);
    glVertex3f(-sx, sy, -sz);
    glVertex3f(sx, sy, -sz);
    glVertex3f(sx, -sy, -sz);
    // top
    glNormal3f( 0.0f, 0.0f,1.0f);
    glVertex3f(-sx, -sy, sz);
    glVertex3f(-sx, sy, sz);
    glVertex3f(sx, sy, sz);
    glVertex3f(sx, -sy, sz);
    // back
    glNormal3f(-1.0f, 0.0f, 0.0f);
    glVertex3f(-sx, -sy, -sz);
    glVertex3f(-sx, sy, -sz);
    glVertex3f(-sx, sy, sz);
    glVertex3f(-sx, -sy, sz);
    // front
    glNormal3f( 1.0f, 0.0f, 0.0f);
    glVertex3f(sx, -sy, -sz);
    glVertex3f(sx, sy, -sz);
    glVertex3f(sx, sy, sz);
    glVertex3f(sx, -sy, sz);
    // left
    glNormal3f( 0.0f, -1.0f, 0.0f);
    glVertex3f(-sx, -sy, -sz);
    glVertex3f(sx, -sy, -sz);
    glVertex3f(sx, -sy, sz);
    glVertex3f(-sx, -sy, sz);
    //right
    glNormal3f( 0.0f, 1.0f, 0.0f);
    glVertex3f(-sx, sy, -sz);
    glVertex3f(sx, sy, -sz);
    glVertex3f(sx, sy, sz);
    glVertex3f(-sx, sy, sz);
    glEnd();
}

void CViewerBTree::_plotTreeNodesRecursive( const CBNode< BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS >* p_pNode,
                                            const Eigen::Vector3d& p_vecPosition,
                                            const uint32_t& p_uAngleDegrees ) const
{
    //ds check if this node has leafs
    if( p_pNode->bHasLeaves )
    {
        //ds we have leafs - draw this node as a branch (grey) and dispatch function on leafs
        glColor4f( 0.5, 0.5, 0.5, 0.25 );
        glVertex3d( p_vecPosition.x( ), p_vecPosition.y( ), p_vecPosition.z( ) );

        //ds leaf positions
        Eigen::Vector3d vecPositionLeafSetBit( Eigen::Vector3d::Zero( ) );
        Eigen::Vector3d vecPositionLeafUnsetBit( Eigen::Vector3d::Zero( ) );

        //ds spacing factor
        const double dSpacing = 0.1*p_pNode->uDepth;

        //ds z is directly proportional to the depth value
        vecPositionLeafSetBit.z( )   = 4*dSpacing;
        vecPositionLeafUnsetBit.z( ) = vecPositionLeafSetBit.z( );

        //ds check if we have to fork in x or y
        if( 0 == p_uAngleDegrees%2 )
        {
            //ds fork in x
            vecPositionLeafSetBit.x( )   = p_vecPosition.x( );
            vecPositionLeafUnsetBit.x( ) = p_vecPosition.x( );

            //ds same y coordinate for the split
            vecPositionLeafSetBit.y( )   = p_vecPosition.y( )+dSpacing;
            vecPositionLeafUnsetBit.y( ) = p_vecPosition.y( )-dSpacing;
        }
        else
        {
            //ds fork in y
            vecPositionLeafSetBit.y( )   = p_vecPosition.y( );
            vecPositionLeafUnsetBit.y( ) = p_vecPosition.y( );

            //ds same y coordinate for the split
            vecPositionLeafSetBit.x( )   = p_vecPosition.x( )+dSpacing;
            vecPositionLeafUnsetBit.x( ) = p_vecPosition.x( )-dSpacing;
        }

        //ds dispatch function
        _plotTreeNodesRecursive( p_pNode->pLeafOnes, vecPositionLeafSetBit, p_uAngleDegrees+1 );
        _plotTreeNodesRecursive( p_pNode->pLeafZeros, vecPositionLeafUnsetBit, p_uAngleDegrees+1 );
    }
    else
    {
        //ds no leafs - draw this node as a leaf (GREEN) and terminate recursion
        glColor3f( 0.0, 1.0, 0.0 );
        glVertex3d( p_vecPosition.x( ), p_vecPosition.y( ), p_vecPosition.z( ) );
    }


}

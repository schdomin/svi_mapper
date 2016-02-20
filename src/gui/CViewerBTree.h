#ifndef CVIEWERSCENE_H
#define CVIEWERSCENE_H

#include <QGLViewer/qglviewer.h>
#include <QMatrix4x4>
#include <vector>

//ds custom
#include "../types/CKeyFrame.h"
#include "../types/CBITree.h"



class CViewerBTree: public QGLViewer
{

public:

    CViewerBTree( const std::shared_ptr< CBITree< MAXIMUM_DISTANCE_HAMMING, BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS > > p_pRoot );
    ~CViewerBTree( );

protected:

    virtual void draw( );
    //virtual void keyPressEvent( QKeyEvent* p_pEvent );
    virtual void init( );
    virtual QString helpString( ) const;

public:

    //ds draw current tree
    void manualDraw( );

    //ds recompute query nodes: mimics the internal matching function to determine matches
    void highlightQUERY( const std::shared_ptr< const std::vector< CDescriptorBRIEF< DESCRIPTOR_SIZE_BITS > > > p_vecDescriptorsQUERY, const UIDKeyFrame& p_IDQUERY );

private:

    //ds snippet of g2o: opengl_primitives
    void _drawBox( GLfloat l, GLfloat w, GLfloat h ) const;

    //ds plot tree nodes
    void _plotTreeNodesRecursive( const CBNode< BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS >* p_pNode,
                                  const Eigen::Vector3d& p_vecPosition,
                                  const uint32_t& p_uAngleDegrees ) const;

private:

    //ds active readonly tree handle
    const std::shared_ptr< CBITree< MAXIMUM_DISTANCE_HAMMING, BTREE_MAXIMUM_DEPTH, DESCRIPTOR_SIZE_BITS > > m_pBTree;

    //ds last query point holder: <position, matching success>
    std::vector< std::pair< Eigen::Vector3d, bool > > m_vecNodesQUERY;

    //ds display
    GLUquadricObj* m_pQuadratic;
};

#endif //CVIEWERSCENE_H

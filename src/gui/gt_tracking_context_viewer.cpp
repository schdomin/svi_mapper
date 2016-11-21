#include "gt_tracking_context_viewer.h"
#include "opengl_primitives.h"
#include <iostream>

using namespace std;

namespace gtracker {

  using namespace gl_primitives;

  TrackingContextViewer::TrackingContextViewer(const std::shared_ptr< std::vector< CKeyFrame* > > keyframes_): _keyframes(keyframes_) {
    _frames_drawn=true;
    _landmarks_drawn=true;
    _follow_robot=true;
  }

  void TrackingContextViewer::drawFrame(const CKeyFrame* keyframe_) {
    Eigen::Matrix4f global_T = keyframe_->matTransformationLEFTtoWORLD.matrix().cast<float>();
    global_T.row(3) << 0,0,0,1;
    glPushMatrix();
    glMultMatrixf(global_T.data());
    glPushAttrib(GL_LINE_WIDTH|GL_POINT_SIZE);
    glColor3f(0.5, 0.5, 0.8);
    glLineWidth(0.1);
    drawPyramidWireframe(1, 1);
    glPopMatrix();
  }


  /*void TrackingContextViewer::drawLandmarks() {
    glPointSize(3);
    glBegin(GL_POINTS);
    for(LandmarkPtrMap::iterator it=_context->landmarks().begin(); it!=_context->landmarks().end(); it++){
      Landmark* l=it->second;
      gt_real base_intensity=0.3;
      gt_real active_intensity=0.8;
      gt_real size=1;
      gt_real active_size=5;
      Vector3 depth_color(0, 0, 1);
      Vector3 vision_color(1, 0, 0);
      Vector3 closure_color(0, 1, 0);
      
      glPushAttrib(GL_COLOR|GL_POINT_SIZE);
      Vector3 color=depth_color;
      //gt_real point_size=size;

      if (l->isByVision())
        color=vision_color;
      else
        color=depth_color;

      if (0<_correspondences.count(l->index())) {
        size  = active_size;
        color = closure_color;
      } else if (l->isActive()) {
        size=active_size;
        color=depth_color;
	      color*=active_intensity;
      } else {
        color=vision_color;
        color*=base_intensity;
      }
      glPointSize(size);
      glColor3f(color.x(), color.y(), color.z());
	
      glVertex3f(l->coordinates().x(), l->coordinates().y(), l->coordinates().z());
      glPopAttrib();
    }
    glEnd();
  }*/

  void TrackingContextViewer::draw(){
    Eigen::Matrix4f world_to_robot;
    world_to_robot.setIdentity();
    if(_follow_robot) {
        if (0 < _keyframes->size()) {
            world_to_robot=_keyframes->back()->matTransformationLEFTtoWORLD.inverse().matrix().cast<float>();
        }
    }
    world_to_robot.matrix().row(3) << 0,0,0,1;

    glPushMatrix();
    glMultMatrixf(world_to_robot.data());
    
    //ds add frames
    glPointSize(2.0);
    glBegin(GL_POINTS);
    for (const CPoint3DWORLD& frame_pose_: _frames) {
        glColor3f(0.0,0,1.0);
        glVertex3f(frame_pose_.x(), frame_pose_.y(), frame_pose_.z());
    }
    glEnd();

    //ds add keyframes
    for (const CKeyFrame* keyframe: *_keyframes){
      glPushAttrib(GL_POINT_SIZE|GL_COLOR);
      glPointSize(1);
      glColor3f(0.5,0,0);
      drawFrame(keyframe);
      glPopAttrib();
    }
    
    if (_landmarks_drawn)
      //drawLandmarks();
    glPopMatrix();
  }

}

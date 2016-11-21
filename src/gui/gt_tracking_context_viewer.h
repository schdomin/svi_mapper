#pragma once
#include "simple_viewer.h"
#include "../types/CKeyFrame.h"

namespace gtracker {

  class TrackingContextViewer: public SimpleViewer{
  public:
    TrackingContextViewer(const std::shared_ptr< std::vector< CKeyFrame* > > keyframes_);

    inline bool landmarksDrawn() const {return _landmarks_drawn;}
    inline void setLandmarksDrawn(bool landmarks_drawn) {_landmarks_drawn=landmarks_drawn;}
    
    inline bool framesDrawn() const {return _frames_drawn;}
    inline void setFramesDrawn(bool frames_drawn) {_frames_drawn=frames_drawn;}

    inline bool followRobot() const {return _follow_robot;}
    inline void setFollowRobot(bool follow_robot) {_follow_robot=follow_robot;}

    void add(const CPoint3DWORLD& frame_pose_) {_frames.push_back(frame_pose_);}
    void updateLandmarks(const std::vector<CLandmark*>& landmarks_) {_landmarks=landmarks_;}

  protected:
    
    virtual void draw();

    void drawFrame(const CKeyFrame* keyframe_);
    //void drawLandmarks();
    bool _frames_drawn;
    bool _landmarks_drawn;
    bool _follow_robot;

    const std::shared_ptr<std::vector<CKeyFrame*>> _keyframes;
    std::vector<CPoint3DWORLD> _frames;
    std::vector<CLandmark*> _landmarks;
  };

}

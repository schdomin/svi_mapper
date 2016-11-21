#include "simple_viewer.h"
#include <cstring>
#include <Eigen/Geometry>

namespace gtracker {
  using namespace std;
  using namespace Eigen;
  
  class StandardCamera: public qglviewer::Camera {
  public:
    StandardCamera(): _standard(true) {}
  
    float zNear() const {
      if(_standard) { return 0.001f; } 
      else { return Camera::zNear(); } 
    }

    float zFar() const {  
      if(_standard) { return 10000.0f; } 
      else { return Camera::zFar(); }
    }

    bool standard() const { return _standard; }  
    void setStandard(bool s) { _standard = s; }

  protected:
    bool _standard;
  };


  QKeyEvent* SimpleViewer::lastKeyEvent() {
    if (_last_key_event_processed)
      return 0;
    return &_last_key_event;
  }
  
  void SimpleViewer::keyEventProcessed() {
    _last_key_event_processed = true;
  }
  SimpleViewer::SimpleViewer() :
    _last_key_event(QEvent::None, 0, Qt::NoModifier){
    _last_key_event_processed = true;
  }

  void SimpleViewer::keyPressEvent(QKeyEvent *e) {
    QGLViewer::keyPressEvent(e);
    _last_key_event = *e;
    _last_key_event_processed=false;
  }

  void SimpleViewer::init() {
    // Init QGLViewer.
    QGLViewer::init();
    // Set background color light yellow.
    // setBackgroundColor(QColor::fromRgb(255, 255, 194));

    
    // Set background color white.
    setBackgroundColor(QColor::fromRgb(255, 255, 255));

    // Set some default settings.
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_BLEND); 
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);
    glShadeModel(GL_FLAT);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Don't save state.
    setStateFileName(QString::null);

    // Mouse bindings.
    setMouseBinding(Qt::RightButton, CAMERA, ZOOM);
    setMouseBinding(Qt::MidButton, CAMERA, TRANSLATE);
    setMouseBinding(Qt::ControlModifier, Qt::LeftButton, RAP_FROM_PIXEL);

    // Replace camera.
    qglviewer::Camera *oldcam = camera();
    qglviewer::Camera *cam = new StandardCamera();
    setCamera(cam);
    cam->setPosition(qglviewer::Vec(0.0f, 0.0f, 0.0f));
    cam->setUpVector(qglviewer::Vec(0.0f, -1.0f, 0.0f));
    cam->lookAt(qglviewer::Vec(0.0f, 0.0f, 1.0f));
    delete oldcam;

  }
}

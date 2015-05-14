#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

namespace fs = boost::filesystem;

constexpr auto off = 16;

int main(int argc, char** argv){

  cv::CascadeClassifier cascade_;
  if(!cascade_.load("/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml")){
    std::cout << "Fail to load haarcascade_frontalface_alt.xml" << std::endl;
    return -1;
  }

  BOOST_FOREACH(const fs::path& dir, std::make_pair(fs::directory_iterator("./src/"), fs::directory_iterator())){
    if(fs::is_directory(dir)){
      fs::create_directory("./dst/" + dir.stem().string());
      BOOST_FOREACH(const fs::path& img_path, std::make_pair(fs::directory_iterator(dir), fs::directory_iterator())){
        if(!fs::is_directory(img_path)){
          auto origin_ = cv::imread(img_path.string());

          std::vector<cv::Rect> faces_;
          cascade_.detectMultiScale(origin_, faces_);
          if(!faces_.empty()){
            auto width = origin_.size().width;
            auto height = origin_.size().height;

            for(auto i=0; i!=faces_.size(); i++){
              int tlx = std::max(faces_[i].tl().x - faces_[i].size().width / off, 0);
              int tly = std::max(faces_[i].tl().y - faces_[i].size().height / off, 0);
              int brx = std::min(faces_[i].br().x + faces_[i].size().width / off, width);
              int bry = std::min(faces_[i].br().y + faces_[i].size().height / off, height);
              cv::Rect area(tlx, tly, brx - tlx, bry - tly);
              cv::Mat face_(origin_, area);
              imwrite("./dst/" + dir.stem().string() + "/" + img_path.stem().string() + "_" + boost::lexical_cast<std::string>(i)+ ".jpg", face_);
            }
          }
        }
      }
    }
  }
}
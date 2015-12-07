SIMPLY
	$ cmake . && make
        $ ln -s /home/"$USER"/Downloads testVideo
        $ cd ~/Downloads/ && wget <AVAILABLE_SRC>/oriVideo.mov && cd -

BEFORE COMMIT
        $ make clean && mv CMakeLists.txt ../ && rm -rf *make* && rm -rf *Make* && rm -rf testVideo && rm -f outputFrame/*png && mv ../CMakeLists.txt ./

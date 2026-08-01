##taskkill /FI "WindowTitle eq assetsServer*" /T /F
cd tools/assetsServer
start "assetsServer" assetsServer-win.exe -i ../../assets -wsPort 8041 -httpPort 8040 -wsHost 127.0.0.1  -distWebViewer 
exit
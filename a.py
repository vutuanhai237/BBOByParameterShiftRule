import moviepy.editor as mp

clip = mp.VideoFileClip("a.gif")
clip.write_videofile("a.mp4")
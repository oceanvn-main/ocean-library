import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
ax = fig.add_subplot(111)

# Normal figure text (causes crash if animated)
text_crash = fig.text(0.5, 0.9, "I crash")

# Axes text but with Figure transform (hack)
text_hack = ax.text(0.5, 0.8, "I work", transform=fig.transFigure)

print(text_crash.axes)
print(text_hack.axes)

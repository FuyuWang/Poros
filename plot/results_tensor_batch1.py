import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'Times New Roman'

import numpy as np

plt.figure(figsize=(24., 2))

x = np.arange(5)*0.5
total_width, n = 0.42, 6
width = total_width / n
dist = 1.

plt.subplot(1,2,1)
rl = np.array([1., 1., 1., 1., 1.])
hasco = rl / [4, 4, 2.7, 4.2, 3.1]
confuciux = rl / [6, 5, 3.5, 5, 4.5]
ga = rl / np.array([9.1, 6.6, 1.7, 3.6, 5.3])
bo = rl / np.array([7.3, 6.5, 4.1, 6.3, 6.0])
anneal = rl / np.array([8.1, 5.5, 2.6, 7.4, 5.9])

plt.grid(linestyle='--',linewidth=1.5, axis='y', zorder=0)
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.75)
ax.spines['left'].set_linewidth(1.75)
ax.spines['right'].set_linewidth(1.75)
ax.spines['top'].set_linewidth(1.75)

plt.bar(x + 0 * width*dist, anneal, width=width, color='#DAE3F3', hatch='xxx', zorder=100, label='SA')
plt.bar(x + 1 * width*dist, bo, width=width, color='#DAE3F3', hatch='///', zorder=100, label='BO')
plt.bar(x + 2 * width*dist, ga, width=width, color='#B4C7E7', hatch='xxx', zorder=100, label='GA')
plt.bar(x + 3 * width*dist, confuciux, width=width, color='#B4C7E7', hatch='///', zorder=100, label='ConfuciuX')
plt.bar(x + 4 * width*dist, hasco, width=width, color='#8FAADC', hatch='xxx', zorder=100, label='HASCO')
plt.bar(x + 5 * width*dist, rl, width=width, color='#8FAADC', hatch='|||', zorder=100, label='Poros')
plt.xticks([])
plt.tick_params(axis='both', labelsize=16)
font = {'family': 'Times New Roman',  'style': 'normal', 'weight': 'bold', 'size': 16}
plt.legend(ncol=6, loc='center', fontsize=16, prop=font, bbox_to_anchor=(1.05, 1.1), frameon=False)


plt.subplot(1,2,2)
rl = np.array([1., 1., 1., 1., 1.])
hasco = rl / [1.1, 2.9, 1.6, 1.8, 2.2]
confuciux = rl / np.array([1.3, 3.2, 1.8, 2.8, 2.6])
ga = rl / np.array([1.9, 5.,  3.8, 3.5, 3.5])
bo = rl / np.array([2.1, 4.5, 3.6, 3.9, 3.5])
anneal = rl / np.array([1.8, 4.7, 3.4, 3.7, 3.4])

plt.grid(linestyle='--',linewidth=1.5, axis='y', zorder=0)
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.75)
ax.spines['left'].set_linewidth(1.75)
ax.spines['right'].set_linewidth(1.75)
ax.spines['top'].set_linewidth(1.75)

plt.bar(x + 0 * width*dist, anneal, width=width, color='#DAE3F3', hatch='xxx', zorder=100, label='SA')
plt.bar(x + 1 * width*dist, bo, width=width, color='#DAE3F3', hatch='///', zorder=100, label='BO')
plt.bar(x + 2 * width*dist, ga, width=width, color='#B4C7E7', hatch='xxx', zorder=100, label='GA')
plt.bar(x + 3 * width*dist, confuciux, width=width, color='#B4C7E7', hatch='///', zorder=100, label='ConfuciuX')
plt.bar(x + 4 * width*dist, hasco, width=width, color='#8FAADC', hatch='xxx', zorder=100, label='HASCO')
plt.bar(x + 5 * width*dist, rl, width=width, color='#8FAADC', hatch='|||', zorder=100, label='Poros')
plt.xticks([])
plt.tick_params(axis='both', labelsize=16)
# font = {'family': 'Times New Roman',  'style': 'normal', 'weight': 'bold', 'size': 16}
# plt.legend(ncol=6, loc='center', fontsize=16, prop=font, bbox_to_anchor=(0.5, 1.1), frameon=False)

plt.subplots_adjust(wspace=0.07)
# plt.savefig('./batch1_performance.eps', bbox_inches='tight', pad_inches=0.02)
plt.show()

# plt.subplot(1,2,1)
# rl = np.array([1., 1., 1., 1., 1.])
# hasco = rl * [1.7, 1.6, 1.3, 1.4, 1.5]
# confuciux = rl * np.array([2.2, 1.9, 1.7, 1.8, 1.9])
# ga = rl * np.array([4.8, 5.8, 3.5, 5.1, 4.8])
# bo = rl * np.array([5.2, 6.7, 3.4, 5.7, 5.3])
# anneal = rl * np.array([4.3, 4.2, 4.5, 4.8, 4.5])
#
#
# plt.grid(linestyle='--',linewidth=1.5, axis='y', zorder=0)
# ax=plt.gca()
# ax.spines['bottom'].set_linewidth(1.75)
# ax.spines['left'].set_linewidth(1.75)
# ax.spines['right'].set_linewidth(1.75)
# ax.spines['top'].set_linewidth(1.75)
# ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
#
# plt.bar(x + 0 * width*dist, anneal, width=width, color='#DAE3F3', hatch='xxx', zorder=100, label='SA')
# plt.bar(x + 1 * width*dist, bo, width=width, color='#DAE3F3', hatch='///', zorder=100, label='BO')
# plt.bar(x + 2 * width*dist, ga, width=width, color='#B4C7E7', hatch='xxx', zorder=100, label='GA')
# plt.bar(x + 3 * width*dist, confuciux, width=width, color='#B4C7E7', hatch='///', zorder=100, label='ConfuciuX')
# plt.bar(x + 4 * width*dist, hasco, width=width, color='#8FAADC', hatch='xxx', zorder=100, label='HASCO')
# plt.bar(x + 5 * width*dist, rl, width=width, color='#8FAADC', hatch='|||', zorder=100, label='Poros')
# plt.xticks([])
# plt.tick_params(axis='both', labelsize=16)
# # plt.legend(ncol=6, loc='center', fontsize=16, bbox_to_anchor=(0.5, 1.1), frameon=False)
#
# plt.subplot(1,2,2)
# rl = np.array([1., 1., 1., 1., 1.])
# hasco = rl * [1.5, 1.4, 1.1, 1.2, 1.2]
# confuciux = rl * (np.array([1.9, 1.8, 1.5, 1.6, 1.6]))
# ga = rl * np.array([2.1, 1.5, 1.1, 2.3, 1.8])
# bo = rl * np.array([2.4, 1.5, 1.1, 1.3, 1.6])
# anneal = rl * np.array([2.0, 1.6, 1.3, 1.6, 1.7])
#
# plt.grid(linestyle='--',linewidth=1.5, axis='y', zorder=0)
# ax=plt.gca()
# ax.spines['bottom'].set_linewidth(1.75)
# ax.spines['left'].set_linewidth(1.75)
# ax.spines['right'].set_linewidth(1.75)
# ax.spines['top'].set_linewidth(1.75)
# ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
#
# plt.bar(x + 0 * width*dist, anneal, width=width, color='#DAE3F3', hatch='xxx', zorder=100, label='SA')
# plt.bar(x + 1 * width*dist, bo, width=width, color='#DAE3F3', hatch='///', zorder=100, label='BO')
# plt.bar(x + 2 * width*dist, ga, width=width, color='#B4C7E7', hatch='xxx', zorder=100, label='GA')
# plt.bar(x + 3 * width*dist, confuciux, width=width, color='#B4C7E7', hatch='///', zorder=100, label='ConfuciuX')
# plt.bar(x + 4 * width*dist, hasco, width=width, color='#8FAADC', hatch='xxx', zorder=100, label='HASCO')
# plt.bar(x + 5 * width*dist, rl, width=width, color='#8FAADC', hatch='|||', zorder=100, label='Poros')
# plt.xticks([])
# plt.tick_params(axis='both', labelsize=16)
#
# plt.subplots_adjust(wspace=0.07)
# # plt.savefig('./batch1_power.eps', bbox_inches='tight', pad_inches=0.02)
# plt.show()

# plt.subplot(1,2,1)
# rl = np.array([1., 1., 1., 1., 1.])
# hasco = rl * [1.5, 1.7, 1.4, 1.2, 1.5]
# confuciux = rl * np.array([2.3, 2.7, 2.5, 2.0, 2.4])
# ga = rl * np.array([5.5, 6.1, 5., 5.4, 5.5])
# bo = rl * np.array([5., 4.2, 3.5, 5.5, 4.6])
# anneal = rl * np.array([5.3, 5.9, 4.8, 3.6, 4.9])
#
# plt.grid(linestyle='--',linewidth=1.5, axis='y', zorder=0)
# ax=plt.gca()
# ax.spines['bottom'].set_linewidth(1.75)
# ax.spines['left'].set_linewidth(1.75)
# ax.spines['right'].set_linewidth(1.75)
# ax.spines['top'].set_linewidth(1.75)
#
# plt.bar(x + 0 * width*dist, anneal, width=width, color='#DAE3F3', hatch='xxx', zorder=100, label='SA')
# plt.bar(x + 1 * width*dist, bo, width=width, color='#DAE3F3', hatch='///', zorder=100, label='BO')
# plt.bar(x + 2 * width*dist, ga, width=width, color='#B4C7E7', hatch='xxx', zorder=100, label='GA')
# plt.bar(x + 3 * width*dist, confuciux, width=width, color='#B4C7E7', hatch='///', zorder=100, label='ConfuciuX')
# plt.bar(x + 4 * width*dist, hasco, width=width, color='#8FAADC', hatch='xxx', zorder=100, label='HASCO')
# plt.bar(x + 5 * width*dist, rl, width=width, color='#8FAADC', hatch='|||', zorder=100, label='Poros')
# plt.xticks([])
# plt.tick_params(axis='both', labelsize=16)
# # plt.legend(ncol=6, loc='center', fontsize=16, bbox_to_anchor=(0.5, 1.1), frameon=False)
#
# plt.subplot(1,2,2)
# rl = np.array([1., 1., 1., 1., 1.])
# hasco = rl * [1.2, 2.1, 1.1, 1.5, 1.5]
# confuciux = rl * np.array([1.5, 2.4, 1.4, 1.8, 1.8])
# ga = rl * np.array([1.1, 3, 1.1, 4, 2.3])
# bo = rl * np.array([1.6, 5, 1.0, 1.1, 2.1])
# anneal = rl * np.array([1.5, 3.2, 1.2, 2.2, 2.0])
#
# plt.grid(linestyle='--',linewidth=1.5, axis='y', zorder=0)
# ax=plt.gca()
# ax.spines['bottom'].set_linewidth(1.75)
# ax.spines['left'].set_linewidth(1.75)
# ax.spines['right'].set_linewidth(1.75)
# ax.spines['top'].set_linewidth(1.75)
# ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
#
# plt.bar(x + 0 * width*dist, anneal, width=width, color='#DAE3F3', hatch='xxx', zorder=100, label='SA')
# plt.bar(x + 1 * width*dist, bo, width=width, color='#DAE3F3', hatch='///', zorder=100, label='BO')
# plt.bar(x + 2 * width*dist, ga, width=width, color='#B4C7E7', hatch='xxx', zorder=100, label='GA')
# plt.bar(x + 3 * width*dist, confuciux, width=width, color='#B4C7E7', hatch='///', zorder=100, label='ConfuciuX')
# plt.bar(x + 4 * width*dist, hasco, width=width, color='#8FAADC', hatch='xxx', zorder=100, label='HASCO')
# plt.bar(x + 5 * width*dist, rl, width=width, color='#8FAADC', hatch='|||', zorder=100, label='Poros')
# plt.xticks([])
# plt.tick_params(axis='both', labelsize=16)
#
#
# plt.subplots_adjust(wspace=0.07)
# plt.savefig('./batch1_area.eps', bbox_inches='tight', pad_inches=0.02)
# plt.show()

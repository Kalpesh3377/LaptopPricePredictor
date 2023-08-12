# # import requests
# # URL = "https://www.geeksforgeeks.org/data-structures/"
# # r = requests.get(URL)
# # print(r.content)
# #This will not run on online IDE
# import requests
# from bs4 import BeautifulSoup
#
# URL = "https://www.geeksforgeeks.org/data-structures/"
# r = requests.get(URL)
#
# soup = BeautifulSoup(r.content, 'html5lib') # If this line causes an error, run 'pip install html5lib' or install html5lib
# print(soup.prettify())


import matplotlib.pyplot as plt
values = [5,8,9,4,1,6,7,2,3,8]
ax = plt.axes()
# ax.set_xlim([0,50])
# ax.set_ylim([-10,10])
ax.set_xticks([0,5,10,15,20,25,30,35,40,45,50])
ax.set_yticks([-10,-8,-6,-4,-2,0,2,4,6,8,10])
ax.grid()
plt.plot(range(1,11),values)
plt.show()
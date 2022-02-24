'''
@爬取京东格兰仕的商品信息
'''

from urllib import request
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver import ActionChains
from selenium.webdriver.support import expected_conditions as EC
import time
from pyquery import PyQuery as pq
import re
import os
import csv
import requests
import urllib
csv_file = "./jd.csv"

class JD_Spider:

    def __init__(self,item_name):
        url = 'https://www.jd.com/' # 登录网址
        #url = 'https://mall.jd.com/index-1000003367.html?from=pc'
        self.url = url
        self.item_name = item_name

     

        options = webdriver.ChromeOptions() # 谷歌选项

        # 设置为开发者模式，避免被识别
        options.add_experimental_option('excludeSwitches',
                                        ['enable-automation'])
        self.browser  = webdriver.Chrome(executable_path= r"C:\Users\古风云\AppData\Local\Google\Chrome\Application\chromedriver.exe",
                                         options = options)
        self.wait   =  WebDriverWait(self.browser,2)


    def run(self):
        """登陆接口"""
        self.browser.get(self.url)

            # 这里设置等待：等待输入框
            # login_element = self.wait.until(
            #     EC.presence_of_element_located((By.CSS_SELECTOR, 'div.input-plain-wrap > .fm-text')))

        input_edit = self.browser.find_element(By.CSS_SELECTOR,'#key')
        input_edit.clear()
        input_edit.send_keys(self.item_name)


        search_button = self.browser.find_element(By.CSS_SELECTOR,'#search > div > div.form > button')
        search_button.click()# 点击
        time.sleep(2)

        html = self.browser.page_source # 获取 html
        self.parse_html(html)
        current_url = self.browser.current_url # 获取当前页面 url
        initial_url = str(current_url).split('&pvid')[0]

        #保存csv文件
        headers = ['name','price','commit','type']
        with open(csv_file,"w")as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
        


        for i in range(1,50):
            try:
                print('正在解析----------------{}图片'.format(str(i)))
                next_page_url = initial_url + '&page={}&s={}&click=0'.format(str(i*2+1),str(i*60+1))
                #print(next_page_url)
                self.browser.get(next_page_url)

                self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,'#J_goodsList > ul > li')))
                html = self.browser.page_source
                self.parse_html(html)# 对 html 网址进行解析
                time.sleep(2) # 设置频率
            except Exception as e:
                print('Error Next page',e)
                


    def parse_html(self,html):
        doc = pq(html)
        items = doc('#J_goodsList .gl-warp .gl-item').items()
        for item in items:
            try:
                product = {
                    'name': item.find('div > div.p-name > a > em').text().replace('\n','\t') if item.find('div > div.p-name > a > em').text() else "None",
                    'price': item.find('.p-price i').text().replace('\n','\t') if item.find('div > div.p-price > strong > i').text() else "None",
                    'commit': item.find('div > div.p-commit > strong > a').text().replace('\n','\t') if item.find('div > div.p-commit > strong > a').text() else "None",
                    'author': item.find('div > div.p-bookdetails > span.p-bi-name > a').text()if item.find('div > div.p-bookdetails > span.p-bi-name > a').text() else "None",
                    'store': item.find('div > div.p-shopnum > a').text().replace(' |', '').replace('\n','\t') if item.find('div > div.p-shopnum > a').text() else "None",
                    'img': str(re.findall('.*?data-lazy-img="(//.*?.jpg)" source-data-lazy-img=.*?',str(item.find('div > div.p-img > a > img')))) or  str(re.findall('.*?"" src=(//.*?)" style="".*?',str(item.find('div > div.p-img > a > img')))) if item.find('div > div.p-img > a > img') else "None"
                }

                #保存图片
                pname = product['name'].replace("\t","").replace(" ","")
                img_dir_name = './image/'+pname+'.jpg'
                
                #headers = {'User-Agent': ' Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36'}
                #req = urllib.request.Request(url=url,headers=headers)
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-Agent','Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36')]
                urllib.request.install_opener(opener)
      
                  
                imageURL = "https:"+product['img'].split("'")[1]
        
                urllib.request.urlretrieve(imageURL,img_dir_name )
                  
                #保存csv
                with open(csv_file,"a")as f:
                    f_csv = csv.writer(f)
                    f_csv.writerows([(product['name'],product['price'],product['commit'],product['name'].split(" ")[-1])])
                  

            except Exception as e:
                print('Error {}'.format(e))
       



if __name__ =='__main__':
    #item_name = '格兰仕微波炉'
    #item_name = '格兰仕电烤箱'
    #item_name = '格兰仕电饭煲'
    #item_name = '格兰仕打蛋器'
    #item_name = '格兰仕电磁炉'
    #item_name = '格兰仕炊具'
    #item_name = '格兰仕破壁机'
    #item_name = '格兰仕电热锅'
    #item_name = '格兰仕电热水壶'
    #item_name = '格兰仕电压力锅'
    #item_name = '格兰仕冰箱'
    #item_name = '格兰仕绞肉机'
    #item_name = '格兰仕洗衣机'
    #item_name = '格兰仕电开水瓶'
    # = '格兰仕电蒸炉'
    #item_name = '格兰仕干衣机'
    #item_name = '格兰仕空调'
    #item_name = '格兰仕料理机'
    #item_name = '格兰仕燃气灶'
    #item_name = '格兰仕吸油烟机'
    #item_name = '格兰仕吸洗碗机'
    item_name = '格兰仕婴童'
  

    
    spider = JD_Spider(item_name)
    spider.run()
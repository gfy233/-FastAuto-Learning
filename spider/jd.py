'''
@爬取京东格兰仕的商品信息
'''
from selenium.webdriver.common.keys import Keys
from urllib import request
import selenium
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
import pandas as pd
csv_file = "./jd.csv"
all_link_list = []
detail_result = './detailed_result'
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

        


        for i in range(1,5):
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
        print(1)
        items = doc('#J_goodsList .gl-warp .gl-item').items()
        for item in items:
            try:
                #print("!!!",item.attr('data-sku'))
                #根据商品sku获得详情页网址
                link = 'https://item.jd.com/'+str(item.attr('data-sku'))+'.html'
                all_link_list.append(link)
                product = {
                    'name': item.find('div > div.p-name > a > em').text().replace('\n','\t') if item.find('div > div.p-name > a > em').text() else "None",
                    'price': item.find('.p-price i').text().replace('\n','\t') if item.find('div > div.p-price > strong > i').text() else "None",
                    'commit': item.find('div > div.p-commit > strong > a').text().replace('\n','\t') if item.find('div > div.p-commit > strong > a').text() else "None",
                    'author': item.find('div > div.p-bookdetails > span.p-bi-name > a').text()if item.find('div > div.p-bookdetails > span.p-bi-name > a').text() else "None",
                    'store': item.find('div > div.p-shopnum > a').text().replace(' |', '').replace('\n','\t') if item.find('div > div.p-shopnum > a').text() else "None",
                    'img': str(re.findall('.*?data-lazy-img="(//.*?.jpg)" source-data-lazy-img=.*?',str(item.find('div > div.p-img > a > img')))) or  str(re.findall('.*?"" src=(//.*?)" style="".*?',str(item.find('div > div.p-img > a > img')))) if item.find('div > div.p-img > a > img') else "None",
                    'sku': item.attr('data-sku'),
                    'link':'https://item.jd.com/'+str(item.attr('data-sku'))+'.html'

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
                with open(csv_file,"a",encoding='utf-8') as f:
                    #print("write",(product['name'],product['price'],product['commit'],product['sku'],product['link']))
                    f_csv = csv.writer(f)
                    f_csv.writerows([(product['name'],product['price'],product['commit'],product['sku'],product['link'])])
                  

            except Exception as e:
                print('Error {}'.format(e))
       


    '''评论爬取'''

    #设置更长的等待时间，因为详情页反爬虫较强

    def open_browser(self):
        '''设置浏览器'''
        # 若下列命令报错，请进入下面链接下载chromedriver然后放置在/user/bin/下即可
        # https://chromedriver.storage.googleapis.com/index.html?path=2.35/
        self.options = webdriver.ChromeOptions()
        self.browser = webdriver.Chrome(options = self.options)
        # 隐式等待：等待页面全部元素加载完成（即页面加载圆圈不再转后），才会执行下一句，如果超过设置时间则抛出异常
        try:
            self.browser.implicitly_wait(50)
        except:
            print("页面无法加载完成，无法开启爬虫操作！")
        # 显式等待：设置浏览器最长允许超时的时间
        self.wait = WebDriverWait(self.browser, 30)


#因为京东会自动折叠掉用户默认评价(即忽略评价)，如果点击查看忽略评价会蹦出新的评论窗口，所以后续代码需要这两个变量帮助爬虫正常进行。
    def init_variable(self):
        '''初始化变量'''
        #self.keyword = search_key # 商品搜索关键词
        self.isLastPage = False # 是否为页末
        self.ignore_page = False # 是否进入到忽略评论页面
        #self.user_agent = user_agent # 用户代理，这里暂不用它
    #爬取详情页
    def parse_JDpage(self):
        try:
            time.sleep(10) # 下拉后等10s
            # 定位元素（用户名，用户等级，用户评分，用户评论，评论创建时间，购买选择，页码）
            user_names = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, '//div[@class="user-info"]')))
            user_levels = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, '//div[@class="user-level"]')))
            user_stars = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, '//div[@class="comment-column J-comment-column"]/div[starts-with(@class, "comment-star")]')))
            comments = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, '//div[@class="comment-column J-comment-column"]/p[@class="comment-con"]')))
            order_infos = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, '//div[@class="comment-item"]//div[@class="order-info"]')))
            if self.ignore_page == False:
                # 如果没进入忽略页
                page_num = self.wait.until(EC.presence_of_element_located((By.XPATH, '//a[@class="ui-page-curr"]')))
            else:
                # 如果进入忽略页
                page_num = self.wait.until(EC.presence_of_element_located((By.XPATH, '//div[@class="ui-dialog-content"]//a[@class="ui-page-curr"]')))
            # 获取元素下的字段值
            user_names = [user_name.text for user_name in user_names]
            user_levels = [user_level.text for user_level in user_levels]
            user_stars = [user_star.get_attribute('class')[-1] for user_star in user_stars]
            create_times = [" ".join(order_infos[0].text.split(" ")[-2:]) for order_info in order_infos]
            order_infos = [" ".join(order_infos[0].text.split(" ")[:-2]) for order_info in order_infos]
            comments = [comment.text for comment in comments]
            page_num = page_num.text
        except selenium.common.exceptions.TimeoutException:
            print('parse_page: TimeoutException 网页超时')
            return (0,0,0,0,0,0,0)
            self.browser.refresh()
            self.browser.find_element(By.XPATH,r'//li[@data-tab="trigger" and @data-anchor="#comment"]').click()
            time.sleep(30)
            user_names, user_levels, user_stars, comments, create_times, order_infos, page_num = self.parse_JDpage()
        except selenium.common.exceptions.StaleElementReferenceException:
            print('turn_page: StaleElementReferenceException 某元素因JS刷新已过时没出现在页面中')
            user_names, user_levels, user_stars, comments, create_times, order_infos, page_num = self.parse_JDpage()
        return user_names, user_levels, user_stars, comments, create_times, order_infos, page_num


    def turn_JDpage(self):
        # 移到页面末端并点击‘下一页’
        try:
            if self.ignore_page == False:
                self.browser.find_element(By.XPATH,r'//a[@class="ui-pager-next" and @clstag]').send_keys(Keys.ENTER)
            else:
                self.browser.find_element(By.XPATH,r'//a[@class="ui-pager-next" and @href="#none"]').send_keys(Keys.ENTER)
            time.sleep(3) # 点击完等3s
            self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(5) # 下拉后等5s
        # 如果找不到元素
        except selenium.common.exceptions.NoSuchElementException:
            if self.ignore_page == False:
                try:
                    # 如果有忽略评论的页面但没进入，则进入继续翻页
                    self.browser.find_element(By.XPATH,r'//div[@class="comment-more J-fold-comment hide"]/a').send_keys(Keys.ENTER)
                    self.ignore_page = True
                    print("有忽略评论的页面")
                except:
                    # 如果没有忽略评论的页面且最后一页
                    print("没有忽略评论的页面")
                    self.ignore_page = True
                    self.isLastPage = True
            else:
                # 如果有忽略评论的页面且到了最后一页
                print("没有忽略评论的页面")
                self.isLastPage = True
        except selenium.common.exceptions.TimeoutException:
            print('turn_page: TimeoutException 网页超时')
            return
            time.sleep(30)
            self.turn_JDpage()
        # 如果因为JS刷新找不到元素，重新刷新
        except selenium.common.exceptions.StaleElementReferenceException:
            print('turn_page: StaleElementReferenceException 某元素因JS刷新已过时没出现在页面中')
            self.turn_JDpage()

    def JDcrawl_detail(self):
        # 初始化参数
        self.init_variable( )
        unfinish_crawls = 0 # 记录因反爬虫而没有完全爬取的商品数

        link_list = pd.read_csv("link_list.csv")
        print(list(link_list.iloc[:,1]) )
        
        # 依次进入到单独的商品链接里去
        for url in list(link_list.iloc[:,1]):
            print("url",url)
            print("./detailed_result/"+url.split(".html")[0].split("/")[-1]+".csv")
            if(os.path.exists("./detailed_result/"+url.split(".html")[0].split("/")[-1]+".csv")):
                print("已爬取过，不重复爬取")
                continue
            #https://item.jd.com/10069334656053.html
            df_user_names = []
            df_user_levels = []
            df_user_stars = []
            df_comments = []
            df_create_times = []
            df_order_infos = []

            # 打开模拟浏览器
            self.open_browser()
            # 进入目标网站
            self.browser.get(url)
            time.sleep(35)
            # 进入评论区
            self.browser.find_element(By.XPATH,r'//li[@data-tab="trigger" and @data-anchor="#comment"]').click()
            time.sleep(15)
            # 开始爬取
            self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            self.isLastPage = False
            self.ignore_page = False
            self.lastpage = 0
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " 开启数据爬虫url:",url)
            while self.isLastPage != True:
                page_num = 0
                user_names, user_levels, user_stars, comments, create_times, order_infos, page_num = self.parse_JDpage()
                # 如果某页因为反爬虫无法定位到‘下一页’元素，导致重复爬取同一页，则保留之前的爬取内容，然后就不继续爬这个商品了
                if self.lastpage != page_num:
                    self.lastpage = page_num
                    print("已爬取完第%s页" % page_num)
                    df_user_names.extend(user_names)
                    df_user_levels.extend(user_levels)
                    df_user_stars.extend(user_stars)
                    df_comments.extend(comments)
                    df_create_times.extend(create_times)
                    df_order_infos.extend(order_infos)
                    self.turn_JDpage()
                else:
                    unfinish_crawls += 1
                    self.browser.quit()
                    break
            # 退出浏览器
            self.browser.quit()

            # 保存结果
            results = pd.DataFrame({'user_names':df_user_names,
                                    'user_levels':df_user_levels,
                                    'user_stars':df_user_stars,
                                    'comments':df_comments,
                                    'create_times':df_create_times,
                                    'order_infos':df_order_infos})
            url_id = url.split('/')[-1].split('.')[0]
            save_path = r'./detailed_result/' + str(url_id) + '.csv'
            results.to_csv(save_path, index = False,encoding='utf-8-sig')
            print("爬虫结束，共%d条数据，结果保存至%s" % (len(results),save_path))

if __name__ =='__main__':
    #item_name = '格兰仕微波炉'
    #item_name = '格兰仕电烤箱'
    item_name = '格兰仕电饭煲'
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
    #item_name = '格兰仕婴童'

    #保存csv文件
    headers = ['name','price','commit','sku','link']
    with open(csv_file,"w") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)

    spider = JD_Spider(item_name)
    #爬取列表页

    # spider.run()
    # pd.DataFrame(all_link_list).to_csv("link_list.csv")

    #爬取详情页
    spider.JDcrawl_detail()

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import numpy as np
import pandas as pd
import re, urllib, random, time
import datetime, html, tqdm, os

import warnings
warnings.filterwarnings('ignore')


class IGbot() :
    def __init__(self, usrname=None, passwrd=None, browser=None, max_comments=1000,
                 since=365, max_posts=500, outdir='outputs', include_movies=True):
        self.usrname = usrname
        self.passwrd = passwrd
        self.max_comments = max_comments
        self.since = since
        self.max_posts = max_posts
        self.outdir = outdir
        self.include_movies = include_movies
        if browser is not None :
            self.browser = browser
        else :
            chrome_options = webdriver.ChromeOptions()
            #chrome_options.add_argument("--headless")
            self.browser = webdriver.Chrome(ChromeDriverManager().install(), 
                                            chrome_options=chrome_options)
        
    
    def sleep_random(self, tmin=1, tmax=5):
        time.sleep(random.randint(tmin, tmax))
        

    def SignIn(self, usrname=None, passwrd=None):
        "Sign in to the IG account with give username and password."
        self.usrname = self.usrname if usrname is None else usrname
        self.passwrd = self.passwrd if passwrd is None else passwrd

        if self.usrname is None or self.passwrd is None :
            raise Exception("username and paddword are not set")

        self.browser.get('https://www.instagram.com/accounts/login/')
        self.sleep_random()
        usrnameInput = self.browser.find_elements_by_css_selector('form input')[0]
        passwrdInput = self.browser.find_elements_by_css_selector('form input')[1]
        
        usrnameInput.send_keys(self.usrname)
        passwrdInput.send_keys(self.passwrd)
        passwrdInput.send_keys(Keys.ENTER)

        return True


    def load_all_comments(self, max_comments) :
        "Loading all comments by pressing 'load more comments' button."
        more_comments = True
        cnt = 0
        while (more_comments == True and cnt <= max_comments/12):
            try :
                d = self.browser.find_element_by_class_name("dCJp8")
                load_button = d.find_element_by_tag_name("span")
                load_button.click()
                cnt += 1
                self.sleep_random(tmin=3, tmax=4)
            except :
                more_comments = False

        
    def get_post_id(self, post) : 
        return post.split(r'/')[-2]

 
    def get_username(self): 
        header = self.browser.find_element_by_class_name("nZSzR")
        return header.find_element_by_tag_name("h2").text

        
    def get_post_likes(self) :
        try :
            likes_span = self.browser.find_element_by_class_name('Nm9Fw')
            likes = int(likes_span.find_element_by_tag_name('span').text.replace(',', ''))
            return likes
        except :
            return np.nan
            

    def get_post_time(self) : 
        dt = self.browser.find_element_by_class_name('_1o9PC').get_attribute('datetime')
        return datetime.datetime.strptime(dt[:19], '%Y-%m-%dT%H:%M:%S')

        
    def public_acc(self):
        return not bool(self.browser.find_elements_by_xpath("//*[contains(text(), 'This Account is Private')]"))


    def register_user_info(self, username=None):
        "Gathering user information such as num. of followers, posts, etc."
        if username is not None :
            self.browser.get(r'https://www.instagram.com/'+ username)
        self.sleep_random()
        user_elements = self.browser.find_elements_by_class_name('g47SY')
        user_metrics_dict = {}
        user_metrics_dict['username'] = self.get_username()
        user_metrics_dict['num_posts'] = user_elements[0].text.replace(',', '')
        user_metrics_dict['followers'] = user_elements[1].get_attribute('title').replace(',', '')
        user_metrics_dict['following'] = user_elements[2].text.replace(',', '')

        return user_metrics_dict


    def list_all_posts_from_user(self, max_posts=1000):
        "Preparing a list of all posts in order to scrape"
        posts = set()
        old_len = -1
        while (len(posts) <= max_posts and old_len < len(posts)) :
            old_len = len(posts)
            links = self.browser.find_elements_by_tag_name('a')
            for link in links :
                post = link.get_attribute('href')
                if '/p/' in post :
                    posts.add(post)
            self.browser.find_element_by_tag_name('html').send_keys(Keys.END)
            self.sleep_random(tmin=3, tmax=5)
                    
        return list(posts)
    
    
    def scrape_all_comments_from_post(self, post_url=None) :
        "Scrape all comments from a given post."
        if post_url is not None :
            self.browser.get(post_url)
        self.load_all_comments(self.max_comments)
        comment_lst = []
        cmnts = self.browser.find_element_by_class_name("XQXOT").find_elements_by_class_name("Mr508")
        for cm in cmnts:
            d = cm.find_element_by_class_name("ZyFrc").find_element_by_tag_name("li")
            d = d.find_element_by_class_name("P9YgZ").find_element_by_tag_name("div")
            d = d.find_element_by_class_name("C4VMK")
            poster = d.find_element_by_tag_name("h3").text
            post = d.find_elements_by_tag_name("span")[-1].text
            comment_lst.append({
                "poster": poster,
                "post": post
            })

        return comment_lst
    

    def gather_all_posts_from_user(self, username):
        """
           Scrape all info from a user (user info, post time, comments, ...)
           and save it in a dataframe.
        """
        self.browser.get(r"https://www.instagram.com/"+ username)
        print('Gather user info...')
        user_info = self.register_user_info()

        print('Gathering a list of all posts...')
        posts = self.list_all_posts_from_user(max_posts=self.max_posts)
        print('Number of posts to scrape: {}'.format(len(posts)))

        print('Scraping info...')
        df = pd.DataFrame()
        for ii in tqdm.tqdm(range(len(posts))) :
            post = posts[ii]
            self.browser.get(post)
            post_time = self.get_post_time()
            post_id = self.get_post_id(post)
            post_age = datetime.datetime.now() - post_time
            if post_age.days <= self.since :
                post_dict = user_info
                post_dict['url'] = post
                post_dict['post_time'] = post_time
                post_dict['post_id'] = post_id
                post_dict['num_likes'] = self.get_post_likes()
                post_dict['comments'] = self.scrape_all_comments_from_post()
                df = df.append(post_dict, ignore_index=True)
            df.to_csv(os.path.join(self.outdir, username+'-df.csv'))        
        return df
    
    
    def save_all_posts_from_list_of_users(self, user_list) :
        "Loop over users in user_list and generate a dataframe for each one."
        if not os.path.isdir(self.outdir) :
            os.makedirs(self.outdir)
        for usrname in user_list:
            print("Scraping {} data".format(usrname))
            df = self.gather_all_posts_from_user(usrname)
            df.to_csv(os.path.join(self.outdir, usrname+'-df.csv'))
            print("Saved a DataFrame in output folder.")
        return True


    ################# Downoading images and videos
    def go_to_next(self) :
        "Click on the botton to go to the next image in a post"
        try :
            next_button = self.browser.find_element_by_class_name("EcJQs").find_element_by_class_name("_6CZji")
            next_button.click()
            return True
        except :
            return False

        
    def renavigation_check(self, url):
        "Go to url if it's not already up."
        if (url is not None and not (url == self.browser.current_url)): 
            self.browser.get(url)
            self.sleep_random(tmin=1, tmax=2)
        url = self.browser.current_url if url is None else url
        return url
   

    def get_current_img_download_links(self) :
        "Grab download link of the current image."
        download_links = []
        sc = self.browser.find_element_by_class_name("_97aPb")
        sc = sc.find_elements_by_class_name("ZyFrc")
        for item in sc :
            link = item.find_element_by_class_name("KL4Bh").find_element_by_tag_name("img")
            link = html.unescape(link.get_attribute("src"))
            download_links.append(link)
        return download_links
   

    def get_current_mov_download_links(self) :
        "Grab download link of the current video."
        download_links = []
        fc = self.browser.find_element_by_class_name("_97aPb").find_elements_by_class_name("_5wCQW")
        for item in fc :
            link = item.find_element_by_tag_name("video")
            link = html.unescape(link.get_attribute("src"))
            download_links.append(link)
        return download_links


    def download_post_items(self, post_url, download_folder) :
        "Download all images/videos from a post with post_url and save it to download_folder"
        self.renavigation_check(post_url)
        download_links = self.get_current_img_download_links()
        while self.go_to_next() :
            dl = self.get_current_img_download_links()
            for item in dl :
                if item not in download_links :
                    download_links.append(item)
            if self.include_movies :
                dl = self.get_current_mov_download_links()
                for item in dl :
                    if item not in download_links :
                        download_links.append(item)

        for cnt, dl in enumerate(download_links) :
            fname = "{}-{}.jpg".format(self.get_post_id(post_url), cnt)
            print(fname)
            save_name = os.path.join(download_folder, fname)
            try:
                urllib.request.urlretrieve(dl, save_name)
            except:
                print("Couldn't save this: {}".format(fname))
        return True

    
    def download_items_from_list_posts(self, post_list) :
        "Loop over all posts in post_list and download all their images/videos"
        download_folder = os.path.join(self.outdir, "downloads")
        if not os.path.isdir(download_folder) :
            os.makedirs(download_folder)
        for url in post_list :
            self.browser.get(url)
            self.download_post_items(url, download_folder)

        return True  
       
    
if __name__ == '__main__':
    username = 'IG_scraper007'
    password = 'howtoscrape101'
    max_posts = 50
    IGhandle = IGbot(username, password, max_posts=max_posts)
    IGhandle.SignIn()

    ## Collect user information
    username = "dodobaror"
    user_info = IGhandle.register_user_info(username=username)
    print(user_info)

    IGhandle.browser.get(r'https://www.instagram.com/'+ username)
    posts = IGhandle.list_all_posts_from_user(max_posts=50)
    print("Number of posts scraped: {}".format(len(posts)))
    print("\nSome example post links:")
    posts[:5]

    ## Scrape comments from one post
    post_url = posts[25]
    comm_lst = IGhandle.scrape_all_comments_from_post(post_url=post_url)
    print("\nSome example comments:")
    comm_lst

    ## Scrape comments from all posts of a user
    df = IGhandle.gather_all_posts_from_user(username)
    print(df.info())
    df.head(4)

    ## Download links from one post
    IGhandle.browser.get(post_url)
    img_download_links = IGhandle.get_current_img_download_links()
    mov_download_links = IGhandle.get_current_mov_download_links()
    print("Image download links:\n")
    print(img_download_links)
    print("Movie download links:\n")
    print(mov_download_links)

    ## Download images and videos from one post
    IGhandle.download_post_items(post_url, "outputs/downloads/")

    ## Download all images and videos from scraped posts of a user
    user_list = ['dodobaror',
                 'katyperry',
                 'kyliejenner',
                 'dresslikejayda',
    ]

    IGhandle.save_all_posts_from_list_of_users(user_list)
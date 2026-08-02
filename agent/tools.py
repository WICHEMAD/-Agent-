import urllib.parse
from optparse import Option
from pydantic import Field
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from langchain_classic.schema import document
from langchain_core.tools import Tool, BaseTool
from sqlalchemy.engine import default
from sqlalchemy.ext.asyncio import result
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

#==========================计算器工具==========================
class CalculatorTool(BaseTool):
    name:str= "simple_calculator"
    description:str= (
            "当求职者问及薪资计算、涨薪比例、税后收入等数学问题时使用。"
            "输入应为一个数学表达式字符串，例如 '8000*1.1'、'10000*0.8'。"
            "返回计算结果。"
)

    def _run(self, query:str)->str:
        try:
            # ✅ 安全检查：只允许数字和基本运算符
            allowed_chars = set("0123456789.+-*/()%^ ")

            #检测非法的字符
            for char in query:
                if char not in allowed_chars:
                    return f"输入格式错误：包含非法字符 {char}"

            #限制表达式长度
            if len(query) > 100:
                return "输入格式错误：表达式长度超过100个字符"

            #执行计算
            result = eval(query, {"__builtins__": {}})
            if isinstance(result, float) and result==int(result):
                result=int(result)
            if isinstance(result, float):
                result=round(result,4)
            return f"计算结果：{result}"

        except SyntaxError:
            return f"错误：表达式语法错误：{query}"
        except ZeroDivisionError:
            return "错误：除数不能为零"
        except Exception as e:
            return f"计算失败：{e}"

    async def _arun(self,query:str):
        return self._run(query)


#=========================岗位爬虫工具==========================
class JobScraperTool(BaseTool):
    name:str= "job_info_scraper"
    description:str= (
        "当求职者需要了解真实岗位信息、薪资行情、岗位要求时使用。"
        "输入为搜索关键词，如 'Python 后端 深圳'。"
        "返回职位信息，包含职位名称、职位描述、职位链接、职位发布时间、职位薪资、职位城市、职位公司名称、职位公司规模、职位公司行业、职位公司地址、职位公司描述。"
    )

    def _run(self, keywords:str)->str:
        options=Options()
        # options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        options.binary_location = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        driver=None

        try:
            driver = webdriver.Chrome(options=options)
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            encoded=urllib.parse.quote(keywords) #URL编码，处理中文、空格、防止链接失效
            url=f"https://www.liepin.com/zhaopin/?key={encoded}"
            driver.get(url)
            print("访问的地址：",url)
            WebDriverWait(driver,15).until(
                EC.presence_of_element_located((By.CLASS_NAME,"job-card-pc-container") )
            )
            time.sleep(1.8)
            cards=driver.find_elements(By.CLASS_NAME,"job-card-pc-container")[:5]
            if not cards:
                print("没有找到岗位卡片元素")
                return _mock_jobs(keywords)

            lines=[f"搜索关键词：{keywords}", ""]
            for i, card in enumerate(cards, 1):
                try:
                    title=card.find_element(By.CLASS_NAME,"job-title-box").text.strip()
                except Exception:
                    title="未知"
                try:
                    salary = card.find_element(By.CLASS_NAME, "job-salary").text.strip()
                except Exception:
                    salary = "薪资面议"
                try:
                    company = card.find_element(By.CLASS_NAME, "company-name").text.strip()
                except Exception:
                    company = "未知公司"

                lines.append(f"{i}. {title} | {salary} | {company}")

            return "\n".join(lines)
        except Exception:
            return _mock_jobs(keywords)
        finally:
            if driver is not None:
                driver.quit()
    async def _arun(self,keywords:str)->str:
        return self._run(keywords)



def _mock_jobs(keywords: str) -> str:
    return (
        f"[模拟数据 - 搜索关键词: {keywords}]\n"
        "1. Python后端开发工程师 | 25k-40k | 某科技有限公司 | 深圳\n"
        "2. 高级Python开发 | 30k-50k | 某某互联网公司 | 北京\n"
        "3. Python数据分析师 | 20k-35k | 数据科技公司 | 上海\n"
        "4. Python全栈工程师 | 22k-38k | 创业科技公司 | 杭州\n"
        "5. Python测试开发 | 18k-30k | 某金融科技公司 | 深圳"
    )
if __name__=="__main__":
    calc=CalculatorTool()
    print(calc._run("8000*1.1"))

    job_scraper=JobScraperTool()
    print(job_scraper._run("Python 后端 深圳"))



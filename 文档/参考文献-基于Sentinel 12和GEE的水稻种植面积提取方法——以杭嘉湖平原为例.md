基于Sentinel 1/2和GEE的水稻种植面积提取方法——以杭嘉湖平原为例





引用格式：鄂海林, 周德成, 李坤. 基于Sentinel 1/2和GEE的水稻种植面积提取方法——以杭嘉湖平原为例\[J]. 智慧农业(中英文), 2025, 7(2): 81-94.



DOI： 10.12133/j.smartag.SA202502003



E Hailin, ZHOU Decheng, LI Kun. Extracting Method of the Cultivation Aera of Rice Based on Sentinel-1/2 and Google Earth Engine (GEE): A Case Study of the Hangjiahu Plain\[J]. Smart Agriculture, 2025, 7(2): 81-94.



DOI： 10.12133/j.smartag.SA202502003



图片

官网全文免费阅读



图片

图片

知网阅读



图片













基于Sentinel 1/2和GEE的水稻种植面积提取方法——以杭嘉湖平原为例







鄂海林1，2， 周德成1\*， 李坤3，4



（1.南京信息工程大学 生态与应用气象学院，江苏南京 210044，中国； 2.中科卫星应用德清研究院 全省微波空间智能云计算重点实验室，浙江湖州 313200，中国； 3.中国科学院空天信息创新研究院，北京 100094，中国； 4.中国科学院大学 资源与环境学院，北京 100049，中国）







摘要： 



［目的/意义］水稻是中国的主要作物之一，准确提取水稻面积对保障粮食安全、温室气体排放管理、水资源调配及生态保护至关重要。光学与微波遥感数据融合是水稻监测主要发展趋势，但现有研究大多依赖传统的物候学特征（如移栽期），忽视了植被和水体指数在水稻生长全过程中的整体动态变化特征。为了快速、准确地获取水稻种植分布、面积等信息，以中国典型水稻种植区—杭嘉湖平原为例，研发了一种基于Sentinel-1/2数据和Google Earth Engine（GEE）云计算平台的水稻种植面积提取方法，即NDVI-SDWI 动态融合水稻识别方法（Dynamic NDVI-SDWI Fusion Method for Rice Mapping, DNSF-Rice）。



［方法］首先，通过Sentinel-2归一化植被指数（Normalized Difference Vegetation Index, NDVI）时间序列，基于阈值分割获取水稻种植潜在分布范围；其次，通过Sentinel-1双极化水体指数（Sentinel-1 Dual-Polarized Water Index, SDWI）时间序列，分析其在水稻生长周期内的动态变化特征，构建阈值分割算法获取基于微波数据的水稻种植分布；最后，将上述结果的交集作为最终水稻分布范围，构建了杭嘉湖平原2019—2023年10 m空间分辨率的水稻种植分布图。此外，利用地面实测数据和统计数据对提取结果进行了精度验证，并与其他产品进行了对比分析。



［结果与讨论］本研究所提取的水稻种植分布图总体精度均达96%以上，F1得分超过0.96，水稻种植面积整体呈逐年增长的趋势，提取面积与统计数据具有高度的一致性，优于其他相关产品。



［结论］DNSF-Rice水稻识别方法基于GEE云平台，结合了光学和合成孔径雷达（Synthetic Aperture Radar, SAR）时间序列数据的优势，利用了NDVI和SDWI在水稻生长全过程中的整体动态变化特征，为高效、精确监测水稻种植面积提供了新的思路。







关键词： 遥感；GEE；种植面积提取；Sentinel-1；合成孔径雷达；归一化植被指数



图片

图片

文章图片











图片

注：该图基于自然资源部标准地图服务网站下载的审图号为GS（2024）0650号标准地图制作，底图无修改。



图 1　杭嘉湖平原位置与野外采样点空间分布



Fig. 1  Spatial distribution of field sampling sites and the location of the Hangjiahu Plain



图片

图2　基于DNSF-Rice方法水稻识别提取技术流程图



Fig. 2  Technical workflow for rice identification and extraction based on the DNSF-Rice method



图片

图3　杭嘉湖平原水稻种植分布研究5 种土地类型的ΔNDVI值分布



Fig. 3  Distribution of ΔNDVI values for five land cover types in the study of rice planting distribution in the Hangjiahu Plain



图片

图4　杭嘉湖平原水稻种植分布研究不同类型地物的SDWI随时间变化曲线



Fig. 4  Temporal variation of SDWI across different land cover types in the study of rice planting distribution in the Hangjiahu Plain



图片

图5　不同地物类型在水稻生长期间的SDWI值分布



Fig. 5  Distribution of SDWI values for different land cover types during the rice growth period



图片

图6　不同地物类型SDWI>-0.4频率分布



Fig. 6  Frequency distribution of SDWI > -0.4 across different land cover types



图片

注：该图基于自然资源部标准地图服务网站下载的审图号为GS（2024）0650号标准地图制作，底图无修改。



图7　2023年杭嘉湖平原水稻种植分布



Fig. 7  Rice cultivation distribution of Hangjiahu Plain in 2023



图8　DNSF-Rice方法提取水稻面积在县级尺度下与统计面积对比、与不同水稻制图产品对比



图片

Fig. 8  County-level rice area extraction using the DNSF-Rice method： Validation against official statistics and comparison with existing rice maps



图片

注：该图基于自然资源部标准地图服务网站下载的审图号为GS（2024）0650号标准地图制作，底图无修改。



图9　杭嘉湖平原6个典型位置的空间分布



Fig. 9  Spatial distribution of the six selected representative locations in the Hangjiahu Plain



图片

注：虚线框指局部细节区域。



图10 比较位置a—位置f的水稻制图与已发布数据产品结果



Fig. 10  Comparison of rice mapping results for locations a to Locations f with published data products



图片

图11　2019—2023年杭嘉湖地区水稻种植面积



Fig. 11  Rice cultivation area in the Hangjiahu region from 2019 to 2023


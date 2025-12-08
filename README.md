# Pytorch+Minimind实现大模型训练推理
代码总体实现参照项目：

https://github.com/Wood-Q/MokioMind

https://www.bilibili.com/video/BV1T2k6BaEeC?spm_id_from=333.788.videopod.episodes&vd_source=8c0f8eb174e31ee2e9d9a85d1b1f6138&p=5

感谢大佬的开发流程分享！

## Minimind结构图：
<img width="1800" height="1124" alt="Pasted image 20251207114428" src="https://github.com/user-attachments/assets/31246fb1-1bdc-49ba-b695-8c15b32a8f6d" />

## Attention注意力机制
Q：query
K：key
V：value
一维情况下：
<img width="3250" height="1274" alt="ffedd6b9d6901c7b1f95732fbe29c306" src="https://github.com/user-attachments/assets/01fb8e10-d403-448f-95cf-d192da0fa227" />

二维情况下：
注意力分数a(q,ki)可以使用以下几种：
<img width="1012" height="338" alt="9a18b44f35765967939ce9242a7e7e5c" src="https://github.com/user-attachments/assets/2d18e15a-e78b-4ac4-b4d1-4b57bc7b442e" />

我们选择点积模型：
<img width="1592" height="1048" alt="8a6cb77b37ce0a122b5a9f6aebfb2c57" src="https://github.com/user-attachments/assets/16bc192c-9386-4934-93c7-ec7641d67e58" />

多头注意力：允许模型同时关注来自不同位置，不同表示子空间的信息。
注意力头数量h 将dmodel的维度拆分给h个独立的头
最后用concat平均地拼接起来 再乘以W^o

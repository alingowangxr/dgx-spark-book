import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'DGX Spark 玩透指南',
  description: '個人 AI 超級電腦全面實戰 — 從開箱到 AI Agent，一步一步打造你的個人 AI 工作站',
  base: '/dgx-spark-book/',
  lang: 'zh-TW',
  lastUpdated: true,
  cleanUrls: true,

  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }],
    ['meta', { name: 'theme-color', content: '#6c5ce7' }],
  ],

  themeConfig: {
    logo: { src: '/logo.svg', width: 24, height: 24 },
    siteTitle: 'DGX Spark 教學',

    nav: [
      { text: '首頁', link: '/' },
      { text: '開始閱讀', link: '/guide/chapter1/' },
      { text: '模型清單', link: '/guide/models' },
      { text: 'FAQ', link: '/guide/faq' },
    ],

    sidebar: {
      '/guide/': [
        {
          text: '📦 第一篇：硬體與系統建置',
          items: [
            { text: '第 1 章：DGX Spark 硬體總覽', link: '/guide/chapter1/' },
            { text: '第 2 章：DGX OS 安裝與首次開機', link: '/guide/chapter2/' },
            { text: '第 3 章：Linux 環境與 Claude Code', link: '/guide/chapter3/' },
            { text: '第 4 章：遠端桌面與網路存取', link: '/guide/chapter4/' },
          ],
        },
        {
          text: '🧠 第二篇：LLM 推論入門',
          items: [
            { text: '第 5 章：Ollama', link: '/guide/chapter5/' },
            { text: '第 6 章：Open WebUI', link: '/guide/chapter6/' },
            { text: '第 7 章：LM Studio', link: '/guide/chapter7/' },
            { text: '第 8 章：llama.cpp', link: '/guide/chapter8/' },
          ],
        },
        {
          text: '🚀 第三篇：LLM 推論進階',
          items: [
            { text: '第 9 章：vLLM', link: '/guide/chapter9/' },
            { text: '第 10 章：TensorRT-LLM', link: '/guide/chapter10/' },
            { text: '第 11 章：SGLang', link: '/guide/chapter11/' },
            { text: '第 12 章：NIM 與引擎比較', link: '/guide/chapter12/' },
          ],
        },
        {
          text: '🎨 第四篇：多媒體 AI 生成',
          items: [
            { text: '第 13 章：圖片與影片生成', link: '/guide/chapter13/' },
            { text: '第 14 章：音訊、語音與音樂', link: '/guide/chapter14/' },
          ],
        },
        {
          text: '🔧 第五篇：模型微調與訓練',
          items: [
            { text: '第 15 章：LoRA / QLoRA 微調', link: '/guide/chapter15/' },
            { text: '第 16 章：Unsloth 微調', link: '/guide/chapter16/' },
            { text: '第 17 章：LLaMA Factory / NeMo', link: '/guide/chapter17/' },
            { text: '第 18 章：影像模型微調', link: '/guide/chapter18/' },
            { text: '第 19 章：預訓練中小型模型', link: '/guide/chapter19/' },
          ],
        },
        {
          text: '🤖 第六篇：多模態 AI 與智慧代理',
          items: [
            { text: '第 20 章：多模態推論', link: '/guide/chapter20/' },
            { text: '第 21 章：RAG 與知識圖譜', link: '/guide/chapter21/' },
            { text: '第 22 章：AI Agent 與沙箱', link: '/guide/chapter22/' },
          ],
        },
        {
          text: '🔬 第七篇：科學計算與開發工具',
          items: [
            { text: '第 23 章：CUDA-X 與 JAX', link: '/guide/chapter23/' },
            { text: '第 24 章：開發環境與 AI 輔助', link: '/guide/chapter24/' },
            { text: '第 25 章：多機互連', link: '/guide/chapter25/' },
          ],
        },
        {
          text: '📎 附錄',
          items: [
            { text: '附錄 A：Claude Code 指令速查', link: '/guide/appendix-a/' },
            { text: '附錄 B：官方 Playbook 對照', link: '/guide/appendix-b/' },
            { text: '附錄 C：推薦模型清單', link: '/guide/appendix-c/' },
            { text: '附錄 D：常見問題 FAQ', link: '/guide/faq' },
            { text: '附錄 E：硬體規格速查', link: '/guide/appendix-e/' },
          ],
        },
        {
          text: '📋 快速參考',
          items: [
            { text: '推薦模型清單', link: '/guide/models' },
            { text: '常見問題 FAQ', link: '/guide/faq' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/nvidia' },
    ],

    editLink: {
      pattern: 'https://github.com/your-repo/edit/main/docs/:path',
      text: '在 GitHub 上編輯此頁',
    },

    footer: {
      message: '以 CC BY-SA 4.0 授權釋出',
      copyright: 'DGX Spark 玩透指南 — 個人 AI 超級電腦全面實戰',
    },

    search: {
      provider: 'local',
      options: {
        locales: {
          root: {
            translations: {
              button: { buttonText: '搜尋', buttonAriaLabel: '搜尋文件' },
              modal: {
                noResultsText: '找不到相關結果',
                resetButtonTitle: '清除搜尋',
                footer: {
                  selectText: '選擇',
                  navigateText: '切換',
                  closeText: '關閉',
                },
              },
            },
          },
        },
      },
    },

    outline: {
      level: [2, 3],
      label: '本頁目錄',
    },

    docFooter: {
      prev: '上一頁',
      next: '下一頁',
    },

    sidebarMenuLabel: '選單',
    returnToTopLabel: '回到頂端',
    lastUpdated: {
      text: '最後更新於',
    },
  },

  markdown: {
    lineNumbers: true,
  },

  ignoreDeadLinks: true,
})

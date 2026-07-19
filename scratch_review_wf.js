export const meta = {
  name: 'follow-pipeline-review',
  description: '复核5-7月解读正确率 + 对抗审查镜像跟单盈亏方法',
  phases: [
    { title: '解读复核', detail: '批量判每条系统解读对错' },
    { title: '盈亏审查', detail: '对抗审查mirror_follow_pnl方法+抽查' },
    { title: '汇总', detail: '合并' },
  ],
}
const ROOT = '/Users/xin/Documents/Claude/Projects/money/quant_system'

const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    n_reviewed: { type: 'integer' },
    errors: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string', description: '消息id或idx' },
          ts: { type: 'string' },
          text: { type: 'string', description: '原文前80字' },
          system_said: { type: 'string', description: '系统解读/动作' },
          should_be: { type: 'string', description: '正确解读' },
          klass: { type: 'string', description: '错误类别: 漏出场/误买/误噪音/方向错/其他' },
          severity: { type: 'string', enum: ['critical', 'major', 'minor'] },
        },
        required: ['id', 'text', 'system_said', 'should_be', 'klass', 'severity'],
      },
    },
    verdict: { type: 'string', description: '这批的正确率印象 + 主要错误模式(2-3句)' },
  },
  required: ['n_reviewed', 'errors', 'verdict'],
}

phase('解读复核')
const NBATCH = 8
const reviewPrompts = []
for (let i = 0; i < NBATCH; i++) {
  reviewPrompts.push(`工作目录 ${ROOT}。你是 enrich 期权跟单系统的解读审查员(第 ${i + 1}/${NBATCH} 批)。
用 /usr/local/opt/python@3.13/bin/python3.13 读 output/interp_audit.json (一个数组, 每项含 idx/id/ts/text/rule/llm/final_action = 系统对该条enrich消息的最终解读与动作)。
按 idx % ${NBATCH} == ${i} 取你负责的分片。逐条判断: 【系统的final_action是否正确反映了enrich在这条消息里的真实交易意图】。
enrich风格: 买入=$票+到期+行权价+calls/puts+权利金; 出场=自然语言(scale out/down to runners/all cash/trim/cutting/manage to X%/holding 1/N), 常多票一句话; 噪音=watchlist/复盘/喊话/评论/"Closing +5%"(收盘播报非出场)。
重点抓这几类错(项目已知痛点):
  · 漏出场: 他明明在减仓/清仓(尤其多票"down to 1/3 both"、"paying across the board sell into strength"、"cutting $X holding $Y"、持仓百分比更新"holding 1/4"、"manage 35%"), 系统却判成 忽略/仅提醒 → 跟单者跟不出来
  · 误买: 把评论/watchlist当买入信号
  · 方向错: calls/puts判反
  · 误噪音: 真信号被当NOISE
只报错的条目(对的不用列)。每条给 id/ts/text/system_said(读final_action)/should_be/klass/severity。n_reviewed=你实际过了多少条。默认严格但公正: 多票出场只要系统最终有按票执行(final_action含'LLM捞漏出场'或'出场指令')就算对; 只有'忽略'或'仅提醒'才算漏。`)
}
const reviews = await parallel(reviewPrompts.map((p, i) =>
  () => agent(p, { label: `复核批${i + 1}`, phase: '解读复核', schema: REVIEW_SCHEMA, effort: 'high' })
))

phase('盈亏审查')
const AUDIT_SCHEMA = {
  type: 'object',
  properties: {
    trustworthy: { type: 'string', enum: ['yes', 'no', 'partial'] },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['critical', 'major', 'minor'] },
          summary: { type: 'string' },
          evidence: { type: 'string' },
        },
        required: ['severity', 'summary', 'evidence'],
      },
    },
    verdict: { type: 'string' },
  },
  required: ['trustworthy', 'findings', 'verdict'],
}
const auditCommon = `工作目录 ${ROOT}。审查 mirror_follow_pnl.py (纯镜像跟单盈亏重建) 与其输出 output/mirror_follow_pnl.json。项目历史被前视/肥尾均值/美化坑过。用 /usr/local/opt/python@3.13/bin/python3.13 验算, 默认怀疑。`
const auditTasks = [
  { label: '审:方法学/前视', prompt: `${auditCommon}
查: ①入场后出场是否严格用时间戳 e.ts>entry_ts(不算买入前同日出场)? ②BS估值用entry-IV在出场日, 财报IV crush是否让部分乐观? 方向如何? ③"扛到期=剩余到期内在价值"是否合理反映风险? ④估值区间[收盘~强势]是否诚实? ⑤规则vs规则+LLM两套出场, LLM增强版会不会把不该执行的NOISE误当出场→虚高? 抽查cache里几条被判exit的NOISE原文核对。给 trustworthy 与 findings。` },
  { label: '审:统计/肥尾', prompt: `${auditCommon}
查: ①报告用中位数+胜率而非均值(均值肥尾), 对不对? 用 mirror_follow_pnl.json 看收益分布偏度。②"方向胜率"与"收益胜率"含义差异是否讲清? ③lotto vs 波段分档是否揭示了真实风险结构? ④"扛到期%"这个指标是不是本分析最重要的风险信号——若他真有大量单不喊出场, 纯跟单根本做不到"他出我出"? 给 trustworthy 与 findings。` },
  { label: '独立抽查3笔', prompt: `${auditCommon}
不看脚本结论, 独立复算3笔(1个大赢/1个归零/1个扛到期)。从 mirror_follow_pnl.json 挑, 去 enrich_history.json 找该票买入后的所有消息, 人工判断他到底有没有喊出场、几点喊、当时股价方向。用日K(candlesticks Day)核对方向与大致收益。报告你的独立数据点是否支持"规则+LLM比纯规则显著减少扛到期/改善收益"。给 trustworthy 与 findings。` },
]
const audits = await parallel(auditTasks.map(t =>
  () => agent(t.prompt, { label: t.label, phase: '盈亏审查', schema: AUDIT_SCHEMA, effort: 'high' })
))

phase('汇总')
return {
  reviews: reviews.map((r, i) => ({ batch: i + 1, ...(r || { n_reviewed: 0, errors: [], verdict: 'FAILED' }) })),
  audits: auditTasks.map((t, i) => ({ who: t.label, ...(audits[i] || { trustworthy: 'FAILED', findings: [], verdict: '' }) })),
}

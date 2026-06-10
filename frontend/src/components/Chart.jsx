import * as echarts from 'echarts'
import { useEffect, useRef } from 'react'

const BASE = {
  backgroundColor: 'transparent',
  textStyle: { color: '#dce4f5', fontSize: 11.5 },
  grid: { left: 48, right: 20, top: 34, bottom: 28 },
  tooltip: { trigger: 'axis', backgroundColor: '#1c2540', borderColor: '#283353', textStyle: { color: '#dce4f5', fontSize: 11.5 } },
  legend: { textStyle: { color: '#8b97b5', fontSize: 11 }, top: 2, type: 'scroll' },
}

export default function Chart({ option, height = 300, onClick }) {
  const ref = useRef(null)
  const chartRef = useRef(null)
  useEffect(() => {
    const chart = echarts.init(ref.current)
    chartRef.current = chart
    const ro = new ResizeObserver(() => chart.resize())
    ro.observe(ref.current)
    return () => { ro.disconnect(); chart.dispose() }
  }, [])
  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return
    chart.setOption({ ...BASE, ...option }, true)
    chart.off('click')
    if (onClick) chart.on('click', onClick)
  }, [option, onClick])
  return <div ref={ref} style={{ height, width: '100%' }} />
}

export const axisX = (labels) => ({
  type: 'category', data: labels,
  axisLine: { lineStyle: { color: '#283353' } },
  axisLabel: { color: '#8b97b5', interval: 5 },
})
export const axisY = (name) => ({
  type: 'value', name,
  nameTextStyle: { color: '#8b97b5' },
  axisLine: { show: false },
  splitLine: { lineStyle: { color: '#1f2842' } },
  axisLabel: { color: '#8b97b5' },
})

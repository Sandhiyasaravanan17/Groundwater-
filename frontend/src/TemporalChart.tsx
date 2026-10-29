import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

function format(value: number) { return new Intl.NumberFormat('en-IN', { maximumFractionDigits: 2 }).format(value) }

export default function TemporalChart({ data }: { data: { year: number; mean: number | null }[] }) {
  return <ResponsiveContainer width="100%" height="100%"><LineChart data={data} margin={{ top: 12, right: 18, bottom: 4, left: 4 }}>
    <CartesianGrid stroke="#e7e9e2" vertical={false} />
    <XAxis dataKey="year" tickLine={false} axisLine={false} />
    <YAxis tickLine={false} axisLine={false} width={48} />
    <Tooltip formatter={(value) => [format(Number(value)), 'Observed mean']} />
    <Line type="monotone" dataKey="mean" stroke="#22836d" strokeWidth={2.5} dot={{ r: 4, fill: '#fffdf8', stroke: '#22836d', strokeWidth: 2 }} />
  </LineChart></ResponsiveContainer>
}

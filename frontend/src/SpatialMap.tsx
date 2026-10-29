import { useEffect } from 'react'
import { CircleMarker, MapContainer, TileLayer, Tooltip, useMap } from 'react-leaflet'
import type { Point, Spatial } from './App'
import 'leaflet/dist/leaflet.css'

const classColors: Record<string, string> = { Excellent: '#27846d', Good: '#82a64b', Poor: '#d17a3c', 'Very Poor': '#ba584a', 'Unsuitable for Drinking': '#a63d36' }

function FitPoints({ points }: { points: Point[] }) {
  const map = useMap()
  useEffect(() => {
    if (points.length > 1) map.fitBounds(points.map((point) => [point.latitude, point.longitude]), { padding: [22, 22], maxZoom: 12 })
    else if (points.length === 1) map.setView([points[0].latitude, points[0].longitude], 11)
  }, [map, points])
  return null
}

export default function SpatialMap({ data, compact = false }: { data: Spatial | null; compact?: boolean }) {
  if (!data) return <div className="map-empty">Map observations are unavailable.</div>
  const center: [number, number] = data.points.length
    ? [data.points.reduce((sum, point) => sum + point.latitude, 0) / data.points.length, data.points.reduce((sum, point) => sum + point.longitude, 0) / data.points.length]
    : [11.75, 79.75]

  return <div className={`map-wrap${compact ? ' compact-map' : ''}`}>
    <MapContainer center={center} zoom={9} scrollWheelZoom={!compact} preferCanvas className="leaflet-map">
      <TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      <FitPoints points={data.points} />
      {data.points.map((point) => <CircleMarker key={`${point.source_dataset}-${point.source_record_number}`} center={[point.latitude, point.longitude]} radius={compact ? 5 : 6} pathOptions={{ color: '#fffdf8', weight: 1, fillColor: classColors[point.wqi_class] ?? '#89938c', fillOpacity: 0.88 }}>
        <Tooltip><strong>{point.village}</strong><br />{point.block} · {point.year} · {point.season}<br />Observed WQI: {point.wqi.toFixed(1)} · {point.wqi_class}<br />Source label: {point.quality_class_source || 'Unlabeled'}<br />{point.latitude.toFixed(5)}, {point.longitude.toFixed(5)} · Measured</Tooltip>
      </CircleMarker>)}
    </MapContainer>
    <div className="map-status"><i /> Observed records · {data.metadata.selected_records} mapped</div>
  </div>
}

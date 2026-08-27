<template>
  <!--
    An SVG in the source's own pixel coordinates, laid over whatever is showing
    underneath. Because the viewBox is the frame size and preserveAspectRatio
    matches object-fit: contain, the boxes land on the objects at any display
    size with no resize arithmetic anywhere - the same approach the gallery
    thumbnails use.
  -->
  <svg
    v-if="width > 0 && height > 0"
    class="detection-overlay"
    :viewBox="`0 0 ${width} ${height}`"
    preserveAspectRatio="xMidYMid meet"
    aria-hidden="true"
  >
    <g v-for="(item, index) in drawn" :key="index">
      <rect
        :x="item.x"
        :y="item.y"
        :width="item.w"
        :height="item.h"
        :stroke="item.colour"
        fill="none"
        stroke-width="2"
        vector-effect="non-scaling-stroke"
        rx="2"
      />
      <rect
        v-if="showLabels"
        :x="item.x"
        :y="item.labelY"
        :width="item.labelWidth"
        :height="labelHeight"
        :fill="item.colour"
        rx="2"
      />
      <text
        v-if="showLabels"
        :x="item.x + labelHeight * 0.28"
        :y="item.labelY + labelHeight * 0.74"
        :font-size="fontSize"
        fill="#0a0b0f"
        font-weight="600"
      >{{ item.text }}</text>
    </g>
  </svg>
</template>

<script setup>
/**
 * Detection boxes drawn over a video, a webcam feed or a still.
 *
 * Takes coordinates in the source frame's pixels, which is exactly what the
 * detection endpoints return, so nothing between the model and the screen has
 * to know how large the element happens to be rendered.
 */
import { computed } from 'vue'

const props = defineProps({
  /** Frame width in pixels, as the model saw it. */
  width: { type: Number, default: 0 },
  /** Frame height in pixels, as the model saw it. */
  height: { type: Number, default: 0 },
  /** [{ label_id, label_name, score, box: [x1, y1, x2, y2] }] */
  detections: { type: Array, default: () => [] },
  showLabels: { type: Boolean, default: true },
})

// Hue by class position, stepped by the golden angle so neighbouring classes
// never land on a similar colour. The same rule as the annotation gallery, so
// a class keeps its colour from the project page through to a live feed.
const GOLDEN_ANGLE = 137.508

// FNV-1a, for a detection whose class index is missing.
const hashOf = (text) => {
  let hash = 0x811c9dc5
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193) >>> 0
  }
  return hash % 360
}

const colourFor = (labelId, labelName) => {
  const position = Number.isInteger(labelId) && labelId >= 0
    ? labelId
    : hashOf(String(labelName || ''))
  const hue = (position * GOLDEN_ANGLE) % 360
  const lightness = position % 2 === 0 ? 62 : 72
  return `hsl(${hue.toFixed(1)} 90% ${lightness}%)`
}

// Sized against the frame rather than the screen, so a label stays legible on
// a 4K clip and does not swamp a small one.
const fontSize = computed(
  () => Math.max(10, Math.round(Math.min(props.width, props.height) * 0.045)))
const labelHeight = computed(() => Math.round(fontSize.value * 1.45))

const drawn = computed(() => (props.detections || []).map((detection) => {
  const [x1, y1, x2, y2] = detection.box || [0, 0, 0, 0]
  const text = `${detection.label_name ?? 'object'} ${(detection.score ?? 0).toFixed(2)}`
  return {
    x: x1,
    y: y1,
    w: Math.max(1, x2 - x1),
    h: Math.max(1, y2 - y1),
    text,
    // A box at the very top has no room above it for its label, so the label
    // drops inside the box instead of being clipped off the frame.
    labelY: y1 - labelHeight.value >= 0 ? y1 - labelHeight.value : y1,
    labelWidth: Math.min(props.width - x1,
      text.length * fontSize.value * 0.58 + fontSize.value),
    colour: colourFor(detection.label_id, detection.label_name),
  }
}))
</script>

<style scoped>
.detection-overlay {
  position:absolute;
  inset:0;
  width:100%;
  height:100%;
  pointer-events:none;
}
</style>

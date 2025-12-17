package com.livestock.recognition.services

import android.graphics.*
import android.graphics.drawable.BitmapDrawable
import android.graphics.drawable.Drawable
import com.livestock.recognition.data.models.ClassificationResult

/**
 * Service for creating visual annotations on images with bounding boxes and labels.
 * Provides visual markup for classification results with confidence indicators.
 * 
 * Requirements: 5.1
 */
class ImageAnnotationService {
    
    companion object {
        private const val BOUNDING_BOX_STROKE_WIDTH = 8f
        private const val LABEL_TEXT_SIZE = 48f
        private const val LABEL_PADDING = 16f
        private const val CONFIDENCE_BAR_HEIGHT = 12f
        private const val CONFIDENCE_BAR_WIDTH = 200f
        private const val HIGH_CONFIDENCE_THRESHOLD = 0.8f
        private const val MEDIUM_CONFIDENCE_THRESHOLD = 0.6f
    }
    
    /**
     * Creates an annotated image with bounding boxes and classification labels.
     * @param originalBitmap The original image to annotate
     * @param classificationResult The classification result containing breed and confidence information
     * @param boundingBox Optional bounding box coordinates (if null, uses full image)
     * @return Annotated bitmap with visual markup
     */
    fun createAnnotatedImage(
        originalBitmap: Bitmap,
        classificationResult: ClassificationResult,
        boundingBox: RectF? = null
    ): Bitmap {
        // Create a mutable copy of the original bitmap
        val annotatedBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(annotatedBitmap)
        
        // Use full image bounds if no bounding box provided
        val bounds = boundingBox ?: RectF(0f, 0f, originalBitmap.width.toFloat(), originalBitmap.height.toFloat())
        
        // Draw bounding box
        drawBoundingBox(canvas, bounds, classificationResult.breedConfidence)
        
        // Draw classification label
        drawClassificationLabel(canvas, bounds, classificationResult)
        
        // Draw confidence indicator
        drawConfidenceIndicator(canvas, bounds, classificationResult)
        
        return annotatedBitmap
    }
    
    /**
     * Creates a drawable with annotation overlay for use in ImageView.
     * @param originalDrawable The original drawable to annotate
     * @param classificationResult The classification result
     * @param boundingBox Optional bounding box coordinates
     * @return Drawable with annotation overlay
     */
    fun createAnnotatedDrawable(
        originalDrawable: Drawable,
        classificationResult: ClassificationResult,
        boundingBox: RectF? = null
    ): Drawable {
        val bitmap = drawableToBitmap(originalDrawable)
        val annotatedBitmap = createAnnotatedImage(bitmap, classificationResult, boundingBox)
        return BitmapDrawable(null, annotatedBitmap)
    }
    
    /**
     * Draws a bounding box around the detected animal with color-coded confidence.
     */
    private fun drawBoundingBox(canvas: Canvas, bounds: RectF, confidence: Float) {
        val paint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = BOUNDING_BOX_STROKE_WIDTH
            color = getConfidenceColor(confidence)
            pathEffect = null // Solid line for high confidence, could add dashed for low confidence
        }
        
        // Add dashed effect for low confidence
        if (confidence < MEDIUM_CONFIDENCE_THRESHOLD) {
            paint.pathEffect = DashPathEffect(floatArrayOf(20f, 10f), 0f)
        }
        
        canvas.drawRect(bounds, paint)
        
        // Draw corner markers for better visibility
        drawCornerMarkers(canvas, bounds, paint)
    }
    
    /**
     * Draws corner markers at the bounding box corners for better visibility.
     */
    private fun drawCornerMarkers(canvas: Canvas, bounds: RectF, paint: Paint) {
        val markerLength = 30f
        val originalStrokeWidth = paint.strokeWidth
        paint.strokeWidth = originalStrokeWidth * 1.5f
        
        // Top-left corner
        canvas.drawLine(bounds.left, bounds.top, bounds.left + markerLength, bounds.top, paint)
        canvas.drawLine(bounds.left, bounds.top, bounds.left, bounds.top + markerLength, paint)
        
        // Top-right corner
        canvas.drawLine(bounds.right, bounds.top, bounds.right - markerLength, bounds.top, paint)
        canvas.drawLine(bounds.right, bounds.top, bounds.right, bounds.top + markerLength, paint)
        
        // Bottom-left corner
        canvas.drawLine(bounds.left, bounds.bottom, bounds.left + markerLength, bounds.bottom, paint)
        canvas.drawLine(bounds.left, bounds.bottom, bounds.left, bounds.bottom - markerLength, paint)
        
        // Bottom-right corner
        canvas.drawLine(bounds.right, bounds.bottom, bounds.right - markerLength, bounds.bottom, paint)
        canvas.drawLine(bounds.right, bounds.bottom, bounds.right, bounds.bottom - markerLength, paint)
        
        paint.strokeWidth = originalStrokeWidth
    }
    
    /**
     * Draws the classification label with breed name and animal type.
     */
    private fun drawClassificationLabel(canvas: Canvas, bounds: RectF, result: ClassificationResult) {
        val labelText = "${result.breed} (${result.animalType.displayName})"
        
        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = LABEL_TEXT_SIZE
            typeface = Typeface.DEFAULT_BOLD
            isAntiAlias = true
        }
        
        val backgroundPaint = Paint().apply {
            color = getConfidenceColor(result.breedConfidence)
            alpha = 220 // Semi-transparent background
        }
        
        // Measure text dimensions
        val textBounds = Rect()
        textPaint.getTextBounds(labelText, 0, labelText.length, textBounds)
        
        // Calculate label position (above the bounding box)
        val labelLeft = bounds.left
        val labelTop = bounds.top - textBounds.height() - LABEL_PADDING * 2
        val labelRight = labelLeft + textBounds.width() + LABEL_PADDING * 2
        val labelBottom = bounds.top
        
        // Adjust if label would be outside image bounds
        val adjustedLabelTop = if (labelTop < 0) bounds.bottom else labelTop
        val adjustedLabelBottom = if (labelTop < 0) bounds.bottom + textBounds.height() + LABEL_PADDING * 2 else labelBottom
        
        // Draw background rectangle
        val labelRect = RectF(labelLeft, adjustedLabelTop, labelRight, adjustedLabelBottom)
        canvas.drawRoundRect(labelRect, 8f, 8f, backgroundPaint)
        
        // Draw text
        val textX = labelLeft + LABEL_PADDING
        val textY = adjustedLabelTop + textBounds.height() + LABEL_PADDING
        canvas.drawText(labelText, textX, textY, textPaint)
    }
    
    /**
     * Draws a confidence indicator bar showing the prediction confidence.
     */
    private fun drawConfidenceIndicator(canvas: Canvas, bounds: RectF, result: ClassificationResult) {
        val confidenceBarLeft = bounds.left
        val confidenceBarTop = bounds.bottom + LABEL_PADDING
        val confidenceBarRight = confidenceBarLeft + CONFIDENCE_BAR_WIDTH
        val confidenceBarBottom = confidenceBarTop + CONFIDENCE_BAR_HEIGHT
        
        // Draw background bar
        val backgroundPaint = Paint().apply {
            color = Color.GRAY
            alpha = 100
        }
        val backgroundRect = RectF(confidenceBarLeft, confidenceBarTop, confidenceBarRight, confidenceBarBottom)
        canvas.drawRoundRect(backgroundRect, 6f, 6f, backgroundPaint)
        
        // Draw confidence fill
        val confidenceFillWidth = CONFIDENCE_BAR_WIDTH * result.breedConfidence
        val confidencePaint = Paint().apply {
            color = getConfidenceColor(result.breedConfidence)
        }
        val confidenceRect = RectF(confidenceBarLeft, confidenceBarTop, confidenceBarLeft + confidenceFillWidth, confidenceBarBottom)
        canvas.drawRoundRect(confidenceRect, 6f, 6f, confidencePaint)
        
        // Draw confidence percentage text
        val confidenceText = "${(result.breedConfidence * 100).toInt()}%"
        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 32f
            typeface = Typeface.DEFAULT_BOLD
            isAntiAlias = true
            setShadowLayer(2f, 1f, 1f, Color.BLACK)
        }
        
        val textBounds = Rect()
        textPaint.getTextBounds(confidenceText, 0, confidenceText.length, textBounds)
        val textX = confidenceBarLeft + (CONFIDENCE_BAR_WIDTH - textBounds.width()) / 2
        val textY = confidenceBarTop + (CONFIDENCE_BAR_HEIGHT + textBounds.height()) / 2
        canvas.drawText(confidenceText, textX, textY, textPaint)
    }
    
    /**
     * Returns color based on confidence level.
     * High confidence: Green, Medium: Orange, Low: Red
     */
    private fun getConfidenceColor(confidence: Float): Int {
        return when {
            confidence >= HIGH_CONFIDENCE_THRESHOLD -> Color.rgb(76, 175, 80) // Green
            confidence >= MEDIUM_CONFIDENCE_THRESHOLD -> Color.rgb(255, 152, 0) // Orange
            else -> Color.rgb(244, 67, 54) // Red
        }
    }
    
    /**
     * Converts a Drawable to Bitmap for annotation processing.
     */
    private fun drawableToBitmap(drawable: Drawable): Bitmap {
        if (drawable is BitmapDrawable && drawable.bitmap != null) {
            return drawable.bitmap
        }
        
        val bitmap = Bitmap.createBitmap(
            drawable.intrinsicWidth.takeIf { it > 0 } ?: 1,
            drawable.intrinsicHeight.takeIf { it > 0 } ?: 1,
            Bitmap.Config.ARGB_8888
        )
        
        val canvas = Canvas(bitmap)
        drawable.setBounds(0, 0, canvas.width, canvas.height)
        drawable.draw(canvas)
        
        return bitmap
    }
    
    /**
     * Data class representing bounding box coordinates.
     */
    data class BoundingBox(
        val left: Float,
        val top: Float,
        val right: Float,
        val bottom: Float
    ) {
        fun toRectF(): RectF = RectF(left, top, right, bottom)
    }
    
    /**
     * Data class for annotation configuration options.
     */
    data class AnnotationConfig(
        val showBoundingBox: Boolean = true,
        val showLabel: Boolean = true,
        val showConfidenceIndicator: Boolean = true,
        val strokeWidth: Float = BOUNDING_BOX_STROKE_WIDTH,
        val textSize: Float = LABEL_TEXT_SIZE
    )
}
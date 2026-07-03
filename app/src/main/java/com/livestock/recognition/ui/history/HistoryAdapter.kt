package com.livestock.recognition.ui.history

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.livestock.recognition.R
import com.livestock.recognition.core.catalog.BreedNames
import com.livestock.recognition.data.SavedClassification
import com.livestock.recognition.databinding.ItemHistoryBinding
import com.livestock.recognition.image.BitmapLoader
import com.livestock.recognition.ui.common.confidencePercent
import com.livestock.recognition.ui.common.formatDateTime
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class HistoryAdapter(
    private val scope: CoroutineScope,
    private val onClick: (SavedClassification) -> Unit,
    private val onDelete: (SavedClassification) -> Unit,
) : ListAdapter<SavedClassification, HistoryAdapter.ViewHolder>(Diff) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ItemHistoryBinding.inflate(
            LayoutInflater.from(parent.context), parent, false
        )
        return ViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    override fun onViewRecycled(holder: ViewHolder) {
        holder.unbind()
    }

    inner class ViewHolder(
        private val binding: ItemHistoryBinding,
    ) : RecyclerView.ViewHolder(binding.root) {

        private var thumbnailJob: Job? = null

        fun bind(entry: SavedClassification) {
            binding.breedNameText.text = BreedNames.displayName(entry.record.breedLabel)
            binding.metaText.text = formatDateTime(entry.record.capturedAtEpochMillis)
            binding.confidenceText.text = binding.root.context.getString(
                R.string.percent_format,
                confidencePercent(entry.record.confidence),
            )

            binding.root.setOnClickListener { onClick(entry) }
            binding.deleteButton.setOnClickListener { onDelete(entry) }

            binding.thumbnailView.setImageResource(R.drawable.ic_gallery)
            thumbnailJob?.cancel()
            thumbnailJob = scope.launch {
                val bitmap = withContext(Dispatchers.IO) {
                    BitmapLoader.decode(entry.imagePath, THUMBNAIL_DIMENSION)
                }
                if (bitmap != null) {
                    binding.thumbnailView.setImageBitmap(bitmap)
                }
            }
        }

        fun unbind() {
            thumbnailJob?.cancel()
            thumbnailJob = null
        }
    }

    private object Diff : DiffUtil.ItemCallback<SavedClassification>() {
        override fun areItemsTheSame(a: SavedClassification, b: SavedClassification) =
            a.id == b.id

        override fun areContentsTheSame(a: SavedClassification, b: SavedClassification) =
            a == b
    }

    private companion object {
        const val THUMBNAIL_DIMENSION = 128
    }
}

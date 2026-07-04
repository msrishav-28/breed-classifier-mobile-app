package com.livestock.recognition.ui.history

import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.livestock.recognition.R
import com.livestock.recognition.databinding.ActivityHistoryBinding
import com.livestock.recognition.ui.results.ResultsActivity

/**
 * Chronological list of past classifications with per-entry delete and a
 * clear-all action.
 */
class HistoryActivity : AppCompatActivity() {

    private lateinit var binding: ActivityHistoryBinding
    private lateinit var viewModel: HistoryViewModel
    private lateinit var adapter: HistoryAdapter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityHistoryBinding.inflate(layoutInflater)
        setContentView(binding.root)

        viewModel = ViewModelProvider(this)[HistoryViewModel::class.java]

        adapter = HistoryAdapter(
            scope = lifecycleScope,
            onClick = { entry ->
                startActivity(ResultsActivity.savedRecordIntent(this, entry.id))
            },
            onDelete = { entry -> confirmDelete(entry.id) },
        )
        binding.historyList.layoutManager = LinearLayoutManager(this)
        binding.historyList.adapter = adapter

        binding.toolbar.setNavigationOnClickListener { finish() }
        binding.toolbar.setOnMenuItemClickListener { item ->
            if (item.itemId == R.id.action_clear_history) {
                confirmClearAll()
                true
            } else {
                false
            }
        }

        viewModel.entries.observe(this) { entries ->
            adapter.submitList(entries)
            binding.emptyGroup.visibility = if (entries.isEmpty()) View.VISIBLE else View.GONE
            binding.historyList.visibility = if (entries.isEmpty()) View.GONE else View.VISIBLE
        }
    }

    private fun confirmDelete(id: Long) {
        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.delete_entry_title)
            .setMessage(R.string.delete_entry_message)
            .setPositiveButton(R.string.delete) { _, _ -> viewModel.delete(id) }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun confirmClearAll() {
        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.clear_history_title)
            .setMessage(R.string.clear_history_message)
            .setPositiveButton(R.string.delete) { _, _ -> viewModel.clearAll() }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }
}

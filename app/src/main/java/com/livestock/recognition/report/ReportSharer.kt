package com.livestock.recognition.report

import android.content.Context
import android.content.Intent
import androidx.core.content.FileProvider
import com.livestock.recognition.R
import java.io.File

/**
 * Builds a share sheet intent for a generated PDF report via the app's
 * FileProvider.
 */
object ReportSharer {

    private const val MIME_TYPE_PDF = "application/pdf"

    fun shareIntent(context: Context, report: File): Intent {
        val uri = FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            report,
        )
        val send = Intent(Intent.ACTION_SEND).apply {
            type = MIME_TYPE_PDF
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        return Intent.createChooser(send, context.getString(R.string.share_report))
    }
}

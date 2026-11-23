package dev.dubhe.mobile

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ForegroundInfo
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import java.net.ConnectException
import java.net.SocketTimeoutException
import java.net.UnknownHostException
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.CancellationException

internal class ModelDownloadWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
    override suspend fun doWork(): Result {
        val item = mobileCatalog.firstOrNull { it.file == inputData.getString(Model) } ?: return Result.failure()
        val notificationManager = applicationContext.getSystemService(NotificationManager::class.java)
        notificationManager.createNotificationChannel(NotificationChannel(Channel, "模型下载", NotificationManager.IMPORTANCE_LOW))
        val cancel = WorkManager.getInstance(applicationContext).createCancelPendingIntent(id)
        val open = PendingIntent.getActivity(
            applicationContext, 0, Intent(applicationContext, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE,
        )
        fun notification(percent: Int?): Notification = Notification.Builder(applicationContext, Channel)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle(item.name)
            .setContentText(when (percent) {
                null -> "准备下载"
                100 -> "正在校验模型"
                else -> "已下载 $percent%"
            })
            .setProgress(100, percent ?: 0, percent == null || percent == 100)
            .setOnlyAlertOnce(true)
            .setOngoing(true)
            .setContentIntent(open)
            .addAction(Notification.Action.Builder(null, "暂停", cancel).build())
            .build()
        return try {
            setForeground(ForegroundInfo(NoticeId, notification(null), ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC))
            var lastPercent = -1
            ModelStore(applicationContext).download(item) { progress ->
                val percent = (progress * 100).toInt()
                if (percent != lastPercent) {
                    lastPercent = percent
                    setProgressAsync(workDataOf(Percent to percent))
                    notificationManager.notify(NoticeId, notification(percent))
                }
            }
            Result.success()
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (network: Exception) {
            when (network) {
                is UnknownHostException, is SocketTimeoutException, is ConnectException -> {
                    if (runAttemptCount < 3) Result.retry()
                    else Result.failure(workDataOf(Error to "网络连接失败，可以继续下载。"))
                }
                is android.app.ForegroundServiceStartNotAllowedException -> Result.retry()
                else -> Result.failure(workDataOf(Error to (network.localizedMessage ?: "下载失败，可以继续下载。")))
            }
        }
    }

    companion object {
        const val WorkName = "dubhe-model-download"
        const val ModelTag = "model:"
        const val Percent = "percent"
        const val Error = "error"
        private const val Channel = "model-downloads"
        private const val Model = "model"
        private const val NoticeId = 2178

        fun request(item: CatalogModel) = OneTimeWorkRequestBuilder<ModelDownloadWorker>()
            .setInputData(workDataOf(Model to item.file))
            .addTag(ModelTag + item.file)
            .setConstraints(Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build())
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 15, TimeUnit.SECONDS)
            .build()
    }
}

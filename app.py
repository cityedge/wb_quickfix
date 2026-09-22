from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

from PySide6.QtCore import Qt, QTimer, Signal, QPointF, QRectF, QEvent, QSettings
from PySide6.QtGui import QAction, QColor, QCursor, QImage, QKeySequence, QPixmap, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core import (
    EditParams,
    apply_pipeline,
    auto_white_sample,
    sample_median_rgb,
    sample_warning,
    skin_params_from_sample,
    white_params_from_sample,
)


APP_NAME = "WB QuickFix"
APP_VERSION = "1.0.6"
PREVIEW_MAX_DIM = 1800
SAMPLE_SIZE = 11


TEXT = {
    "ja": {
        "open": "開く",
        "save": "名前を付けて保存",
        "undo": "1回戻す",
        "reset": "オールリセット",
        "language": "English",
        "wb": "ホワイトバランス",
        "auto_wb": "自動WB",
        "white_pick": "白を指定",
        "skin_pick": "肌色を指定（実験）",
        "temperature": "色温度",
        "tint": "色偏差",
        "naturalize": "自然補正",
        "basic": "基本補正",
        "reset_wb": "Reset",
        "reset_basic": "Reset",
        "brightness": "明るさ",
        "contrast": "コントラスト",
        "saturation": "彩度",
        "hue": "色相",
        "gamma": "ガンマ",
        "original": "Original",
        "fit": "全体表示",
        "drop": "JPEG / PNG / BMP をここへドロップ\nまたはクリックして開く",
        "status_ready": "画像を開いてください。",
        "status_loaded": "読み込み: {name}  ({w} × {h})",
        "status_white": "白基準: RGB {r}, {g}, {b}  →  色温度 {temp:+d} / 色偏差 {tint:+d} / 自然補正 {nat}",
        "status_auto": "自動WB基準: RGB {r}, {g}, {b}  →  色温度 {temp:+d} / 色偏差 {tint:+d} / 自然補正 {nat}",
        "status_skin": "肌色基準（実験）: RGB {r}, {g}, {b}  →  色温度 {temp:+d} / 色偏差 {tint:+d} / 自然補正 {nat}",
        "sample_dark": "サンプルが暗すぎます。結果が不安定になる可能性があります。",
        "sample_clip": "サンプルが白飛びに近いため、色かぶり情報が少ない可能性があります。",
        "save_ok": "保存しました: {name}",
        "save_error": "保存に失敗しました。\n{error}",
        "open_error": "画像を開けませんでした。\n{error}",
        "need_image": "先にJPEG、PNGまたはBMP画像を開いてください。",
        "skin_help": "肌をクリックすると、複数の肌色基準から近い色度を選び、弱い補正だけを全体に適用します。",
        "white_help": "本来は白またはグレーである場所をクリックしてください。11×11pxの中央値を基準にします。",
        "auto_wb_help": "元画像全体から白・グレーらしい領域を複数抽出し、その色分布の中心を白基準としてホワイトバランスを設定します。画像全体の強い色かぶりも補正対象です。",
        "gamma_tip": "1.0 が無補正です。0 は内部で最小値に置き換えて計算します。",
        "natural_tip": "暗部・ハイライト・過彩度・肌色への過補正をまとめて弱める複合補正です。",
        "output": "保存",
        "output_format": "保存形式",
        "quality": "JPEG品質",
        "png_compression": "PNG圧縮レベル",
        "jpeg_quality_title": "JPEG保存品質",
        "cancel_pick": "スポイト解除",
    },
    "en": {
        "open": "Open",
        "save": "Save As",
        "undo": "Undo Once",
        "reset": "Reset All",
        "language": "日本語",
        "wb": "White Balance",
        "auto_wb": "Auto WB",
        "white_pick": "Pick White",
        "skin_pick": "Pick Skin Tone (Exp.)",
        "temperature": "Temperature",
        "tint": "Tint",
        "naturalize": "Naturalize",
        "basic": "Basic Adjustments",
        "reset_wb": "Reset",
        "reset_basic": "Reset",
        "brightness": "Brightness",
        "contrast": "Contrast",
        "saturation": "Saturation",
        "hue": "Hue",
        "gamma": "Gamma",
        "original": "Original",
        "fit": "Fit",
        "drop": "Drop JPEG / PNG / BMP here\nor click to open",
        "status_ready": "Open an image to begin.",
        "status_loaded": "Loaded: {name}  ({w} × {h})",
        "status_white": "White sample: RGB {r}, {g}, {b}  →  Temperature {temp:+d} / Tint {tint:+d} / Naturalize {nat}",
        "status_auto": "Auto WB sample: RGB {r}, {g}, {b}  →  Temperature {temp:+d} / Tint {tint:+d} / Naturalize {nat}",
        "status_skin": "Skin-tone sample (experimental): RGB {r}, {g}, {b}  →  Temperature {temp:+d} / Tint {tint:+d} / Naturalize {nat}",
        "sample_dark": "The sample is very dark; correction may be unstable.",
        "sample_clip": "The sample is near clipping, so it may contain little color-cast information.",
        "save_ok": "Saved: {name}",
        "save_error": "Could not save the image.\n{error}",
        "open_error": "Could not open the image.\n{error}",
        "need_image": "Open a JPEG, PNG, or BMP image first.",
        "skin_help": "Click skin. The app chooses a nearby chromaticity from several skin references and applies only a conservative global correction.",
        "white_help": "Click an area that should be white or gray. The median of an 11×11 px region is used.",
        "auto_wb_help": "Finds multiple plausible white/gray regions in the untouched source image and uses the robust center of their color distribution as the white reference. Strong global color casts are intentionally corrected toward neutral white.",
        "gamma_tip": "1.0 means no correction. UI value 0 is internally clamped to a small positive number.",
        "natural_tip": "A compound control that moderates shadow/highlight clipping, excess saturation and over-correction of skin-like colors.",
        "output": "Output",
        "output_format": "Format",
        "quality": "JPEG quality",
        "png_compression": "PNG compression",
        "jpeg_quality_title": "JPEG Save Quality",
        "cancel_pick": "Cancel Picker",
    },
}


@dataclass
class AppState:
    params: EditParams

    def clone(self) -> "AppState":
        return AppState(params=self.params.clone())


class ResettableSlider(QSlider):
    editStarted = Signal()
    editFinished = Signal()
    resetRequested = Signal()

    def mousePressEvent(self, event):
        self.editStarted.emit()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.editFinished.emit()

    def mouseDoubleClickEvent(self, event):
        self.resetRequested.emit()
        event.accept()


class ParamControl(QWidget):
    valueChanged = Signal(object)
    editStarted = Signal()
    editFinished = Signal()

    def __init__(
        self,
        key: str,
        minimum: float,
        maximum: float,
        default: float,
        decimals: int = 0,
        step: float = 1.0,
        parent=None,
    ):
        super().__init__(parent)
        self.key = key
        self.default = default
        self.decimals = decimals
        self.scale = 10 ** decimals
        self._syncing = False
        self._spin_edit_snapshot_sent = False

        self.label = QLabel()
        self.label.setMinimumWidth(92)

        self.slider = ResettableSlider(Qt.Horizontal)
        self.slider.setRange(round(minimum * self.scale), round(maximum * self.scale))
        self.slider.setSingleStep(max(1, round(step * self.scale)))
        self.slider.setPageStep(max(1, round(step * self.scale * 5)))
        self.slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        if decimals == 0:
            spin = QSpinBox()
            spin.setRange(int(minimum), int(maximum))
            spin.setSingleStep(max(1, int(step)))
        else:
            spin = QDoubleSpinBox()
            spin.setDecimals(decimals)
            spin.setRange(float(minimum), float(maximum))
            spin.setSingleStep(float(step))
        spin.setFixedWidth(74)
        spin.installEventFilter(self)
        self.spin = spin

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(self.label)
        row.addWidget(self.slider, 1)
        row.addWidget(self.spin)

        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self.slider.editStarted.connect(self.editStarted)
        self.slider.editFinished.connect(self.editFinished)
        self.slider.resetRequested.connect(lambda: self.set_value(self.default, emit=True, begin_edit=True))

        self.set_value(default, emit=False)

    def eventFilter(self, watched, event):
        if watched is self.spin:
            if event.type() == QEvent.FocusIn:
                self._spin_edit_snapshot_sent = False
            elif event.type() in (QEvent.KeyPress, QEvent.Wheel, QEvent.MouseButtonPress):
                if not self._spin_edit_snapshot_sent:
                    self.editStarted.emit()
                    self._spin_edit_snapshot_sent = True
            elif event.type() == QEvent.FocusOut:
                if self._spin_edit_snapshot_sent:
                    self.editFinished.emit()
                self._spin_edit_snapshot_sent = False
        return super().eventFilter(watched, event)

    def _from_slider(self, ivalue: int):
        if self._syncing:
            return
        value = ivalue / self.scale
        self._syncing = True
        self.spin.setValue(value)
        self._syncing = False
        self.valueChanged.emit(value)

    def _from_spin(self, value):
        if self._syncing:
            return
        self._syncing = True
        self.slider.setValue(round(float(value) * self.scale))
        self._syncing = False
        self.valueChanged.emit(float(value) if self.decimals else int(value))

    def set_value(self, value, emit=False, begin_edit=False):
        if begin_edit:
            self.editStarted.emit()
        self._syncing = True
        self.slider.setValue(round(float(value) * self.scale))
        self.spin.setValue(value)
        self._syncing = False
        if emit:
            self.valueChanged.emit(float(value) if self.decimals else int(value))
        if begin_edit:
            self.editFinished.emit()

    def value(self):
        return self.spin.value()

    def set_text(self, text: str):
        self.label.setText(text)


class SplitOriginalItem(QGraphicsPixmapItem):
    """Draw only the right side of the original image during split comparison."""

    def __init__(self):
        super().__init__()
        self._split_x: Optional[float] = None
        self.setZValue(2)
        self.setVisible(False)

    def set_split_x(self, x: Optional[float]):
        self._split_x = None if x is None else float(x)
        self.setVisible(self._split_x is not None and not self.pixmap().isNull())
        self.update()

    def paint(self, painter, option, widget=None):
        if self._split_x is None or self.pixmap().isNull():
            return
        rect = self.boundingRect()
        x = max(rect.left(), min(rect.right(), self._split_x))
        source = QRectF(x, rect.top(), max(0.0, rect.right() - x), rect.height())
        if source.width() > 0.0:
            painter.drawPixmap(source, self.pixmap(), source)

        # A thin cosmetic divider keeps the comparison boundary visible at
        # every zoom level without becoming visually heavy.
        pen = QPen(QColor("#f3f5f8"))
        pen.setWidth(1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))


class ImageView(QGraphicsView):
    sampled = Signal(float, float, str)
    droppedFile = Signal(str)
    emptyClicked = Signal()
    splitCompareStarted = Signal(float)
    splitCompareMoved = Signal(float)
    splitCompareEnded = Signal()
    zoomChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        # QGraphicsView receives most drag/drop input through its viewport.
        # Accept drops on both objects and filter viewport events so local
        # JPEG/PNG/BMP drops work reliably on Windows.
        self.viewport().setAcceptDrops(True)
        self.viewport().installEventFilter(self)
        self.setRenderHints(self.renderHints())
        self.setBackgroundBrush(QColor("#111318"))
        self.setFrameShape(QFrame.NoFrame)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)

        self.scene_obj = QGraphicsScene(self)
        self.setScene(self.scene_obj)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene_obj.addItem(self.pixmap_item)
        self.split_original_item = SplitOriginalItem()
        self.scene_obj.addItem(self.split_original_item)
        self.placeholder = self.scene_obj.addText("")
        self.placeholder.setDefaultTextColor(QColor("#8b93a5"))
        self.placeholder.setZValue(5)

        self.sample_mode: Optional[str] = None
        self._has_image = False
        self._split_dragging = False

    def set_placeholder(self, text: str):
        self.placeholder.setPlainText(text)
        self._center_placeholder()

    def _center_placeholder(self):
        if self._has_image:
            self.placeholder.setVisible(False)
            return
        self.placeholder.setVisible(True)
        rect = self.viewport().rect()
        p = self.mapToScene(rect.center())
        b = self.placeholder.boundingRect()
        self.placeholder.setPos(p.x() - b.width() / 2, p.y() - b.height() / 2)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._center_placeholder()

    def set_pixmap(self, pixmap: QPixmap):
        self.pixmap_item.setPixmap(pixmap)
        self._has_image = not pixmap.isNull()
        self.placeholder.setVisible(not self._has_image)
        self.scene_obj.setSceneRect(self.pixmap_item.boundingRect())
        if not self._has_image:
            self.clear_split_compare()

    def set_original_pixmap(self, pixmap: QPixmap):
        self.split_original_item.setPixmap(pixmap)
        self.split_original_item.set_split_x(None)

    def set_split_position(self, x: float):
        if not self._has_image or self.split_original_item.pixmap().isNull():
            return
        rect = self.pixmap_item.boundingRect()
        self.split_original_item.set_split_x(max(rect.left(), min(rect.right(), float(x))))

    def clear_split_compare(self):
        self._split_dragging = False
        self.split_original_item.set_split_x(None)

    def clear_image(self):
        self.set_pixmap(QPixmap())
        self.split_original_item.setPixmap(QPixmap())
        self.resetTransform()

    def fit_image(self):
        if not self._has_image:
            return
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self.zoomChanged.emit(self.transform().m11())

    def zoom_by(self, factor: float):
        if not self._has_image:
            return
        current = self.transform().m11()
        new = current * factor
        if 0.03 <= new <= 30.0:
            self.scale(factor, factor)
            self.zoomChanged.emit(self.transform().m11())

    def wheelEvent(self, event):
        if not self._has_image:
            super().wheelEvent(event)
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.zoom_by(factor)
        event.accept()

    def set_sample_mode(self, mode: Optional[str]):
        self.sample_mode = mode
        if mode:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.ArrowCursor)

    def mousePressEvent(self, event):
        # With no image loaded, a normal left-click on the image pane behaves
        # exactly like the Open button.
        if not self._has_image and event.button() == Qt.LeftButton:
            self.emptyClicked.emit()
            event.accept()
            return

        # Right-click starts a split comparison at the clicked image position:
        # edited image on the left, untouched original on the right. Holding
        # the button and dragging moves the split boundary horizontally.
        if self._has_image and event.button() == Qt.RightButton:
            p = self.mapToScene(event.position().toPoint())
            rect = self.pixmap_item.boundingRect()
            if rect.contains(p):
                self._split_dragging = True
                self.splitCompareStarted.emit(float(p.x()))
            event.accept()
            return

        if self.sample_mode and self._has_image and event.button() == Qt.LeftButton:
            p = self.mapToScene(event.position().toPoint())
            rect = self.pixmap_item.boundingRect()
            if rect.contains(p):
                self.sampled.emit(p.x(), p.y(), self.sample_mode)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._split_dragging and (event.buttons() & Qt.RightButton):
            p = self.mapToScene(event.position().toPoint())
            rect = self.pixmap_item.boundingRect()
            x = max(rect.left(), min(rect.right(), p.x()))
            self.splitCompareMoved.emit(float(x))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton and self._split_dragging:
            self._split_dragging = False
            self.splitCompareEnded.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        # Right-click is reserved for split Original comparison.
        event.accept()

    def _first_supported_drop_path(self, mime_data) -> Optional[str]:
        if not mime_data or not mime_data.hasUrls():
            return None
        for url in mime_data.urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                if Path(path).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    return path
        return None

    def eventFilter(self, watched, event):
        if watched is self.viewport():
            if event.type() in (QEvent.DragEnter, QEvent.DragMove):
                if self._first_supported_drop_path(event.mimeData()):
                    event.acceptProposedAction()
                    return True
            elif event.type() == QEvent.Drop:
                path = self._first_supported_drop_path(event.mimeData())
                if path:
                    self.droppedFile.emit(path)
                    event.acceptProposedAction()
                    return True
        return super().eventFilter(watched, event)

    def dragEnterEvent(self, event):
        if self._first_supported_drop_path(event.mimeData()):
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event):
        if self._first_supported_drop_path(event.mimeData()):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event):
        path = self._first_supported_drop_path(event.mimeData())
        if path:
            self.droppedFile.emit(path)
            event.acceptProposedAction()
            return
        event.ignore()



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("cityedge", APP_NAME)
        saved_lang = str(self.settings.value("ui/language", "ja"))
        self.lang = saved_lang if saved_lang in ("ja", "en") else "ja"
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1360, 860)
        self.setMinimumSize(980, 620)

        self.state = AppState(EditParams())
        self.previous_state: Optional[AppState] = None
        self._edit_snapshot_active = False
        self._syncing_controls = False
        self._original_hold = False

        self.full_rgb: Optional[np.ndarray] = None
        self.full_alpha: Optional[np.ndarray] = None
        self.preview_rgb: Optional[np.ndarray] = None
        self.preview_alpha: Optional[np.ndarray] = None
        self.current_preview_output: Optional[np.ndarray] = None
        self.image_path: Optional[str] = None
        self.icc_profile = None
        self.exif_bytes = None
        self.dpi = None
        try:
            self.jpeg_quality = int(self.settings.value("output/jpeg_quality", 95))
        except (TypeError, ValueError):
            self.jpeg_quality = 95
        self.jpeg_quality = max(70, min(100, self.jpeg_quality))
        try:
            self.png_compression = int(self.settings.value("output/png_compression", 6))
        except (TypeError, ValueError):
            self.png_compression = 6
        self.png_compression = max(0, min(9, self.png_compression))
        self.output_format = "jpeg"

        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.setInterval(30)
        self.render_timer.timeout.connect(self.render_preview)

        self._build_ui()
        self._build_shortcuts()
        self._apply_dark_theme()
        self.update_language()
        self.update_enabled_state()

    def _build_ui(self):
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(10, 10, 10, 8)
        outer.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # The image pane is intentionally image-only. All controls live in the
        # right pane so the working area stays visually clean.
        self.view = ImageView()
        self.view.sampled.connect(self.on_sampled)
        self.view.droppedFile.connect(self.load_image)
        self.view.emptyClicked.connect(self.open_dialog)
        self.view.splitCompareStarted.connect(self.start_split_compare)
        self.view.splitCompareMoved.connect(self.update_split_compare)
        self.view.splitCompareEnded.connect(self.end_split_compare)
        splitter.addWidget(self.view)

        self.side_scroll = QScrollArea()
        self.side_scroll.setWidgetResizable(True)
        self.side_scroll.setFrameShape(QFrame.NoFrame)
        self.side_scroll.setMinimumWidth(340)
        self.side_scroll.setMaximumWidth(430)
        side = QWidget()
        self.side_layout = QVBoxLayout(side)
        self.side_layout.setContentsMargins(14, 8, 8, 8)
        self.side_layout.setSpacing(8)

        # File / global actions
        action_row = QHBoxLayout()
        action_row.setSpacing(7)
        self.btn_open = QPushButton()
        self.btn_undo = QPushButton()
        self.btn_reset = QPushButton()
        self.btn_open.clicked.connect(self.open_dialog)
        self.btn_undo.clicked.connect(self.undo_once)
        self.btn_reset.clicked.connect(self.reset_all)
        action_row.addWidget(self.btn_open)
        action_row.addWidget(self.btn_undo)
        action_row.addWidget(self.btn_reset)
        self.side_layout.addLayout(action_row)

        # Visual divider between the global actions and white-balance controls.
        self.sep_actions = QFrame()
        self.sep_actions.setFrameShape(QFrame.HLine)
        self.side_layout.addWidget(self.sep_actions)

        wb_header = QHBoxLayout()
        wb_header.setSpacing(8)
        self.lbl_wb = self._section_label()
        self.btn_reset_wb = QPushButton()
        self.btn_reset_wb.setFixedWidth(70)
        self.btn_reset_wb.clicked.connect(self.reset_wb)
        wb_header.addWidget(self.lbl_wb)
        wb_header.addStretch(1)
        wb_header.addWidget(self.btn_reset_wb)
        self.side_layout.addLayout(wb_header)

        self.btn_auto_wb = QPushButton()
        self.btn_auto_wb.clicked.connect(self.apply_auto_wb)
        self.side_layout.addWidget(self.btn_auto_wb)

        pick_row = QHBoxLayout()
        self.btn_white = QPushButton()
        self.btn_skin = QPushButton()
        self.btn_white.setCheckable(True)
        self.btn_skin.setCheckable(True)
        self.btn_white.clicked.connect(lambda checked: self.activate_picker("white", checked))
        self.btn_skin.clicked.connect(lambda checked: self.activate_picker("skin", checked))
        pick_row.addWidget(self.btn_white)
        pick_row.addWidget(self.btn_skin)
        self.side_layout.addLayout(pick_row)

        self.controls = {}
        self.controls["temperature"] = ParamControl("temperature", -200, 200, 0)
        self.controls["tint"] = ParamControl("tint", -100, 100, 0)
        self.controls["naturalize"] = ParamControl("naturalize", 0, 100, 0)
        for key in ("temperature", "tint", "naturalize"):
            self._add_control(self.controls[key])

        self.sep = QFrame()
        self.sep.setFrameShape(QFrame.HLine)
        self.side_layout.addWidget(self.sep)
        basic_header = QHBoxLayout()
        basic_header.setSpacing(8)
        self.lbl_basic = self._section_label()
        self.btn_reset_basic = QPushButton()
        self.btn_reset_basic.setFixedWidth(70)
        self.btn_reset_basic.clicked.connect(self.reset_basic)
        basic_header.addWidget(self.lbl_basic)
        basic_header.addStretch(1)
        basic_header.addWidget(self.btn_reset_basic)
        self.side_layout.addLayout(basic_header)

        self.controls["brightness"] = ParamControl("brightness", -100, 100, 0)
        self.controls["contrast"] = ParamControl("contrast", -100, 100, 0)
        self.controls["saturation"] = ParamControl("saturation", -100, 100, 0)
        self.controls["hue"] = ParamControl("hue", -180, 180, 0)
        self.controls["gamma"] = ParamControl("gamma", 0.0, 3.0, 1.0, decimals=2, step=0.05)
        for key in ("brightness", "contrast", "saturation", "hue", "gamma"):
            self._add_control(self.controls[key])

        self.sep_output = QFrame()
        self.sep_output.setFrameShape(QFrame.HLine)
        self.side_layout.addWidget(self.sep_output)
        self.lbl_output = self._section_label()
        self.side_layout.addWidget(self.lbl_output)

        format_row = QHBoxLayout()
        self.lbl_output_format = QLabel()
        self.combo_output_format = QComboBox()
        self.combo_output_format.setMinimumWidth(120)
        self.combo_output_format.currentIndexChanged.connect(self.on_output_format_changed)
        format_row.addWidget(self.lbl_output_format)
        format_row.addStretch(1)
        format_row.addWidget(self.combo_output_format)
        self.side_layout.addLayout(format_row)

        self.quality_row_widget = QWidget()
        quality_row = QHBoxLayout(self.quality_row_widget)
        quality_row.setContentsMargins(0, 0, 0, 0)
        self.lbl_quality = QLabel()
        self.spin_quality = QSpinBox()
        self.spin_quality.setRange(70, 100)
        self.spin_quality.setValue(self.jpeg_quality)
        self.spin_quality.setFixedWidth(84)
        self.spin_quality.valueChanged.connect(self.on_jpeg_quality_changed)
        quality_row.addWidget(self.lbl_quality)
        quality_row.addStretch(1)
        quality_row.addWidget(self.spin_quality)
        self.side_layout.addWidget(self.quality_row_widget)

        self.png_compression_row_widget = QWidget()
        png_row = QHBoxLayout(self.png_compression_row_widget)
        png_row.setContentsMargins(0, 0, 0, 0)
        self.lbl_png_compression = QLabel()
        self.spin_png_compression = QSpinBox()
        self.spin_png_compression.setRange(0, 9)
        self.spin_png_compression.setValue(self.png_compression)
        self.spin_png_compression.setFixedWidth(84)
        self.spin_png_compression.valueChanged.connect(self.on_png_compression_changed)
        png_row.addWidget(self.lbl_png_compression)
        png_row.addStretch(1)
        png_row.addWidget(self.spin_png_compression)
        self.side_layout.addWidget(self.png_compression_row_widget)

        self.btn_save = QPushButton()
        self.btn_save.clicked.connect(self.save_dialog)
        self.side_layout.addWidget(self.btn_save)

        # View controls also live in the right pane.
        view_row = QHBoxLayout()
        view_row.setSpacing(6)
        self.btn_original = QPushButton()
        self.btn_zoom_out = QToolButton()
        self.btn_zoom_out.setText("−")
        self.btn_zoom_in = QToolButton()
        self.btn_zoom_in.setText("+")
        self.btn_fit = QPushButton()
        self.zoom_label = QLabel("100%")
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setMinimumWidth(52)
        self.view.zoomChanged.connect(lambda z: self.zoom_label.setText(f"{z * 100:.0f}%"))
        self.btn_original.pressed.connect(self.show_original_preview)
        self.btn_original.released.connect(self.restore_edited_preview)
        self.btn_zoom_out.clicked.connect(lambda: self.view.zoom_by(1 / 1.2))
        self.btn_zoom_in.clicked.connect(lambda: self.view.zoom_by(1.2))
        self.btn_fit.clicked.connect(self.view.fit_image)
        view_row.addWidget(self.btn_original, 1)
        view_row.addWidget(self.btn_zoom_out)
        view_row.addWidget(self.zoom_label)
        view_row.addWidget(self.btn_zoom_in)
        view_row.addWidget(self.btn_fit)
        self.side_layout.addLayout(view_row)

        # Language switch stays unobtrusively at the bottom-right, below the
        # view controls.
        self.side_layout.addStretch(1)
        lang_row = QHBoxLayout()
        self.btn_lang = QPushButton()
        self.btn_lang.clicked.connect(self.toggle_language)
        lang_row.addStretch(1)
        lang_row.addWidget(self.btn_lang)
        self.side_layout.addLayout(lang_row)

        self.side_scroll.setWidget(side)
        splitter.addWidget(self.side_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        outer.addWidget(splitter, 1)

        self.setCentralWidget(root)
        self.status = self.statusBar()
        self.status.showMessage("")

    def _section_label(self):
        label = QLabel()
        label.setObjectName("SectionTitle")
        return label

    def _add_control(self, control: ParamControl):
        self.side_layout.addWidget(control)
        control.valueChanged.connect(lambda value, key=control.key: self.on_param_changed(key, value))
        control.editStarted.connect(self.begin_edit_snapshot)
        control.editFinished.connect(self.end_edit_snapshot)

    def _build_shortcuts(self):
        act_open = QAction(self)
        act_open.setShortcut(QKeySequence("Ctrl+O"))
        act_open.triggered.connect(self.open_dialog)
        self.addAction(act_open)

        act_save = QAction(self)
        act_save.setShortcut(QKeySequence("Ctrl+Shift+S"))
        act_save.triggered.connect(self.save_dialog)
        self.addAction(act_save)

        act_undo = QAction(self)
        act_undo.setShortcut(QKeySequence("Ctrl+Z"))
        act_undo.triggered.connect(self.undo_once)
        self.addAction(act_undo)

        act_reset = QAction(self)
        act_reset.setShortcut(QKeySequence("Ctrl+R"))
        act_reset.triggered.connect(self.reset_all)
        self.addAction(act_reset)

        act_escape = QAction(self)
        act_escape.setShortcut(QKeySequence("Esc"))
        act_escape.triggered.connect(self.cancel_picker)
        self.addAction(act_escape)

    def _apply_dark_theme(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #1b1e24; color: #e7e9ee; font-size: 13px; }
            QPushButton, QToolButton {
                background: #2a2f39; border: 1px solid #3b4350; border-radius: 5px;
                padding: 6px 10px; color: #edf0f5;
            }
            QPushButton:hover, QToolButton:hover { background: #343b47; }
            QPushButton:pressed, QToolButton:pressed { background: #222833; }
            QPushButton:checked { background: #3b556e; border-color: #6d98bd; }
            QPushButton:disabled, QToolButton:disabled { color: #6d7480; background: #22262d; }
            QLabel#SectionTitle { font-size: 15px; font-weight: 700; padding-top: 5px; padding-bottom: 3px; }
            QSlider::groove:horizontal { height: 4px; background: #3a404a; border-radius: 2px; }
            QSlider::handle:horizontal { width: 14px; margin: -5px 0; background: #cfd5df; border-radius: 7px; }
            QSlider::sub-page:horizontal { background: #677f98; border-radius: 2px; }
            QSpinBox, QDoubleSpinBox, QComboBox { background: #242932; border: 1px solid #3b4350; border-radius: 4px; padding: 3px; }
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border; width: 18px; background: #3b4553;
                border-left: 1px solid #566170;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-position: top right; border-top-right-radius: 4px;
                border-bottom: 1px solid #566170;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-position: bottom right; border-bottom-right-radius: 4px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover,
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover { background: #556274; }
            QComboBox::drop-down { width: 22px; border-left: 1px solid #566170; background: #3b4553; }
            QComboBox QAbstractItemView { background: #242932; color: #edf0f5; selection-background-color: #3b556e; }
            QScrollArea { background: #1b1e24; }
            QStatusBar { color: #aeb5c2; }
            QSplitter::handle { background: #252a32; width: 2px; }
            QFrame[frameShape="4"] { color: #343a44; }
            """
        )

    def update_language(self):
        t = TEXT[self.lang]
        self.btn_open.setText(t["open"])
        self.btn_save.setText(t["save"])
        self.btn_undo.setText(t["undo"])
        self.btn_reset.setText(t["reset"])
        self.btn_lang.setText(t["language"])
        self.lbl_wb.setText(t["wb"])
        self.btn_auto_wb.setText(t["auto_wb"])
        self.btn_auto_wb.setToolTip(t["auto_wb_help"])
        self.btn_white.setText(t["white_pick"])
        self.btn_skin.setText(t["skin_pick"])
        self.btn_white.setToolTip(t["white_help"])
        self.btn_skin.setToolTip(t["skin_help"])
        self.lbl_basic.setText(t["basic"])
        self.btn_reset_wb.setText(t["reset_wb"])
        self.btn_reset_basic.setText(t["reset_basic"])
        self.lbl_output.setText(t["output"])
        self.lbl_output_format.setText(t["output_format"])
        self.btn_original.setText(t["original"])
        self.btn_fit.setText(t["fit"])
        self.lbl_quality.setText(t["quality"])
        self.lbl_png_compression.setText(t["png_compression"])
        current_format = self.output_format
        self.combo_output_format.blockSignals(True)
        self.combo_output_format.clear()
        self.combo_output_format.addItem("JPEG", "jpeg")
        self.combo_output_format.addItem("PNG", "png")
        self.combo_output_format.addItem("BMP", "bmp")
        idx = self.combo_output_format.findData(current_format)
        self.combo_output_format.setCurrentIndex(max(0, idx))
        self.combo_output_format.blockSignals(False)
        for key, control in self.controls.items():
            control.set_text(t[key])
        self.controls["gamma"].setToolTip(t["gamma_tip"])
        self.controls["naturalize"].setToolTip(t["natural_tip"])
        self.view.set_placeholder(t["drop"])
        if self.full_rgb is None:
            self.status.showMessage(t["status_ready"])
        elif self.image_path:
            h, w = self.full_rgb.shape[:2]
            self.status.showMessage(t["status_loaded"].format(name=Path(self.image_path).name, w=w, h=h))

    def toggle_language(self):
        self.lang = "en" if self.lang == "ja" else "ja"
        self.settings.setValue("ui/language", self.lang)
        self.update_language()

    def on_jpeg_quality_changed(self, value: int):
        self.jpeg_quality = int(value)
        self.settings.setValue("output/jpeg_quality", self.jpeg_quality)

    def on_png_compression_changed(self, value: int):
        self.png_compression = int(value)
        self.settings.setValue("output/png_compression", self.png_compression)

    def update_enabled_state(self):
        has = self.full_rgb is not None
        for w in (
            self.btn_save,
            self.btn_reset,
            self.btn_reset_wb,
            self.btn_reset_basic,
            self.btn_auto_wb,
            self.btn_white,
            self.btn_skin,
            self.btn_original,
            self.btn_zoom_out,
            self.btn_zoom_in,
            self.btn_fit,
            self.spin_quality,
            self.spin_png_compression,
            self.combo_output_format,
        ):
            w.setEnabled(has)
        for c in self.controls.values():
            c.setEnabled(has)

        is_jpeg = has and self.output_format == "jpeg"
        is_png = has and self.output_format == "png"
        self.quality_row_widget.setVisible(is_jpeg)
        self.png_compression_row_widget.setVisible(is_png)
        self.spin_quality.setEnabled(is_jpeg)
        self.lbl_quality.setEnabled(is_jpeg)
        self.spin_png_compression.setEnabled(is_png)
        self.lbl_png_compression.setEnabled(is_png)
        self.btn_undo.setEnabled(has and self.previous_state is not None)

    def open_dialog(self):
        filt = "Images (*.jpg *.jpeg *.png *.bmp);;JPEG (*.jpg *.jpeg);;PNG (*.png);;BMP (*.bmp)"
        path, _ = QFileDialog.getOpenFileName(self, APP_NAME, "", filt)
        if path:
            self.load_image(path)

    def load_image(self, path: str):
        try:
            with Image.open(path) as src:
                self.icc_profile = src.info.get("icc_profile")
                self.dpi = src.info.get("dpi")
                exif = src.getexif()
                img = ImageOps.exif_transpose(src)
                if exif:
                    try:
                        exif[274] = 1  # Orientation after exif_transpose
                        self.exif_bytes = exif.tobytes()
                    except Exception:
                        self.exif_bytes = None
                else:
                    self.exif_bytes = None

                # Palette PNGs can carry transparency via tRNS even though
                # their band list is only ("P",). Preserve that transparency
                # by expanding to RGBA before editing.
                has_alpha = "A" in img.getbands() or (img.mode == "P" and "transparency" in img.info)
                if has_alpha:
                    rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
                    self.full_rgb = rgba[..., :3].copy()
                    self.full_alpha = rgba[..., 3].copy()
                else:
                    self.full_rgb = np.array(img.convert("RGB"), dtype=np.uint8)
                    self.full_alpha = None

            self.image_path = path
            suffix = Path(path).suffix.lower()
            if suffix == ".png":
                self.output_format = "png"
            elif suffix == ".bmp":
                self.output_format = "bmp"
            else:
                self.output_format = "jpeg"
            self._sync_output_format_combo()
            self.state = AppState(EditParams())
            self.previous_state = None
            self._sync_controls_from_state()
            self._build_preview()
            self.view.set_original_pixmap(self._to_pixmap(self.preview_rgb, self.preview_alpha))
            self.render_preview()
            self.view.fit_image()
            self.cancel_picker()
            self.update_enabled_state()
            h, w = self.full_rgb.shape[:2]
            self.status.showMessage(TEXT[self.lang]["status_loaded"].format(name=Path(path).name, w=w, h=h))
        except Exception as exc:
            QMessageBox.critical(self, APP_NAME, TEXT[self.lang]["open_error"].format(error=str(exc)))

    def _build_preview(self):
        h, w = self.full_rgb.shape[:2]
        scale = min(1.0, PREVIEW_MAX_DIM / max(w, h))
        if scale >= 0.999:
            self.preview_rgb = self.full_rgb.copy()
            self.preview_alpha = None if self.full_alpha is None else self.full_alpha.copy()
            return
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        rgb_img = Image.fromarray(self.full_rgb, "RGB").resize((nw, nh), Image.Resampling.LANCZOS)
        self.preview_rgb = np.array(rgb_img, dtype=np.uint8)
        if self.full_alpha is not None:
            a_img = Image.fromarray(self.full_alpha, "L").resize((nw, nh), Image.Resampling.LANCZOS)
            self.preview_alpha = np.array(a_img, dtype=np.uint8)
        else:
            self.preview_alpha = None

    def begin_edit_snapshot(self):
        if self.full_rgb is None or self._edit_snapshot_active:
            return
        self.previous_state = self.state.clone()
        self._edit_snapshot_active = True
        self.update_enabled_state()

    def end_edit_snapshot(self):
        self._edit_snapshot_active = False

    def snapshot_once(self):
        if self.full_rgb is None:
            return
        self.previous_state = self.state.clone()
        self._edit_snapshot_active = False
        self.update_enabled_state()

    def on_param_changed(self, key: str, value):
        if self._syncing_controls or self.full_rgb is None:
            return
        if key == "gamma":
            setattr(self.state.params, key, float(value))
        else:
            setattr(self.state.params, key, int(round(float(value))))
        self.schedule_render()

    def schedule_render(self):
        if self.full_rgb is not None:
            self.render_timer.start()

    def render_preview(self):
        if self.preview_rgb is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            out = apply_pipeline(self.preview_rgb, self.state.params)
            self.current_preview_output = out
            if not self._original_hold:
                self.view.set_pixmap(self._to_pixmap(out, self.preview_alpha))
        finally:
            QApplication.restoreOverrideCursor()

    def _to_pixmap(self, rgb: np.ndarray, alpha: Optional[np.ndarray]) -> QPixmap:
        if alpha is not None:
            rgba = np.dstack([rgb, alpha]).astype(np.uint8, copy=False)
            rgba = np.ascontiguousarray(rgba)
            h, w = rgba.shape[:2]
            qimg = QImage(rgba.data, w, h, rgba.strides[0], QImage.Format_RGBA8888).copy()
        else:
            arr = np.ascontiguousarray(rgb.astype(np.uint8, copy=False))
            h, w = arr.shape[:2]
            qimg = QImage(arr.data, w, h, arr.strides[0], QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def show_original_preview(self):
        if self.preview_rgb is None:
            return
        self.view.clear_split_compare()
        self._original_hold = True
        self.view.set_pixmap(self._to_pixmap(self.preview_rgb, self.preview_alpha))

    def restore_edited_preview(self):
        self._original_hold = False
        if self.current_preview_output is not None:
            self.view.set_pixmap(self._to_pixmap(self.current_preview_output, self.preview_alpha))

    def start_split_compare(self, x: float):
        if self.preview_rgb is None or self.current_preview_output is None:
            return
        self._original_hold = False
        # Ensure the base is the corrected preview; the overlay item reveals
        # the untouched original only to the right of the split boundary.
        self.view.set_pixmap(self._to_pixmap(self.current_preview_output, self.preview_alpha))
        self.view.set_split_position(x)

    def update_split_compare(self, x: float):
        self.view.set_split_position(x)

    def end_split_compare(self):
        self.view.clear_split_compare()

    def apply_auto_wb(self):
        if self.full_rgb is None:
            QMessageBox.information(self, APP_NAME, TEXT[self.lang]["need_image"])
            return

        # Auto WB, like the manual white picker, is always derived from the
        # untouched source image. Existing WB values are replaced, while the
        # user's basic adjustment sliders remain as they are.
        self.snapshot_once()
        self.cancel_picker()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            sample = auto_white_sample(self.full_rgb, self.full_alpha)
            temp, tint, nat = white_params_from_sample(sample)
            self.state.params.temperature = temp
            self.state.params.tint = tint
            self.state.params.naturalize = nat
            self._sync_controls_from_state()

            rgb255 = np.rint(sample * 255).astype(int)
            message = TEXT[self.lang]["status_auto"].format(
                r=int(rgb255[0]), g=int(rgb255[1]), b=int(rgb255[2]),
                temp=temp, tint=tint, nat=nat,
            )
            warn = sample_warning(sample)
            if warn == "too_dark":
                message += "  |  " + TEXT[self.lang]["sample_dark"]
            elif warn == "near_clipped":
                message += "  |  " + TEXT[self.lang]["sample_clip"]
            self.status.showMessage(message)
        finally:
            QApplication.restoreOverrideCursor()

        self.render_preview()
        self.update_enabled_state()

    def activate_picker(self, mode: str, checked: bool):
        if self.full_rgb is None:
            QMessageBox.information(self, APP_NAME, TEXT[self.lang]["need_image"])
            return
        if mode == "white":
            self.btn_skin.blockSignals(True)
            self.btn_skin.setChecked(False)
            self.btn_skin.blockSignals(False)
        else:
            self.btn_white.blockSignals(True)
            self.btn_white.setChecked(False)
            self.btn_white.blockSignals(False)
        # Keep the currently corrected preview visible while picking. Sampling
        # itself still always reads self.full_rgb (the untouched source), so the
        # user can compare point A vs. point B without the view jumping back to
        # Original.
        self.view.set_sample_mode(mode if checked else None)

    def cancel_picker(self):
        self.btn_white.blockSignals(True)
        self.btn_skin.blockSignals(True)
        self.btn_white.setChecked(False)
        self.btn_skin.setChecked(False)
        self.btn_white.blockSignals(False)
        self.btn_skin.blockSignals(False)
        self.view.set_sample_mode(None)
        self.restore_edited_preview()

    def on_sampled(self, px: float, py: float, mode: str):
        if self.full_rgb is None or self.preview_rgb is None:
            return
        ph, pw = self.preview_rgb.shape[:2]
        fh, fw = self.full_rgb.shape[:2]
        x = int(np.clip(round(px * fw / pw), 0, fw - 1))
        y = int(np.clip(round(py * fh / ph), 0, fh - 1))
        sample = sample_median_rgb(self.full_rgb, x, y, SAMPLE_SIZE)
        self.snapshot_once()

        # IMPORTANT: the sample above comes from self.full_rgb, i.e. the
        # untouched source. Picker results REPLACE the visible WB parameters;
        # they are never multiplied into, or derived from, the current preview.
        if mode == "white":
            temp, tint, nat = white_params_from_sample(sample)
            msg = TEXT[self.lang]["status_white"]
        else:
            temp, tint, nat = skin_params_from_sample(sample)
            msg = TEXT[self.lang]["status_skin"]

        self.state.params.temperature = temp
        self.state.params.tint = tint
        self.state.params.naturalize = nat
        self._sync_controls_from_state()

        rgb255 = np.rint(sample * 255).astype(int)
        message = msg.format(
            r=int(rgb255[0]), g=int(rgb255[1]), b=int(rgb255[2]),
            temp=temp, tint=tint, nat=nat,
        )
        warn = sample_warning(sample)
        if warn == "too_dark":
            message += "  |  " + TEXT[self.lang]["sample_dark"]
        elif warn == "near_clipped":
            message += "  |  " + TEXT[self.lang]["sample_clip"]
        self.status.showMessage(message)
        self.cancel_picker()
        self.render_preview()
        self.update_enabled_state()


    def reset_wb(self):
        """Reset only the White Balance group, preserving Basic Adjustments."""
        if self.full_rgb is None:
            return
        self.snapshot_once()
        p = self.state.params
        p.temperature = 0
        p.tint = 0
        p.naturalize = 0
        self._sync_controls_from_state()
        self.cancel_picker()
        self.render_preview()
        self.update_enabled_state()

    def reset_basic(self):
        """Reset only the Basic Adjustments group, preserving white balance."""
        if self.full_rgb is None:
            return
        self.snapshot_once()
        p = self.state.params
        p.brightness = 0
        p.contrast = 0
        p.saturation = 0
        p.hue = 0
        p.gamma = 1.0
        self._sync_controls_from_state()
        self.render_preview()
        self.update_enabled_state()

    def reset_all(self):
        if self.full_rgb is None:
            return
        self.snapshot_once()
        self.state = AppState(EditParams())
        self._sync_controls_from_state()
        self.cancel_picker()
        self.render_preview()
        self.update_enabled_state()

    def undo_once(self):
        if self.previous_state is None or self.full_rgb is None:
            return
        self.state = self.previous_state.clone()
        self.previous_state = None
        self._sync_controls_from_state()
        self.render_preview()
        self.update_enabled_state()

    def _sync_controls_from_state(self):
        self._syncing_controls = True
        try:
            p = self.state.params
            for key, c in self.controls.items():
                c.set_value(getattr(p, key), emit=False)
        finally:
            self._syncing_controls = False

    def _sync_output_format_combo(self):
        if not hasattr(self, "combo_output_format"):
            return
        idx = self.combo_output_format.findData(self.output_format)
        if idx >= 0:
            self.combo_output_format.blockSignals(True)
            self.combo_output_format.setCurrentIndex(idx)
            self.combo_output_format.blockSignals(False)
        self.update_enabled_state()

    def on_output_format_changed(self, _index: int):
        data = self.combo_output_format.currentData()
        if data in ("jpeg", "png", "bmp"):
            self.output_format = data
        self.update_enabled_state()

    def save_dialog(self):
        if self.full_rgb is None:
            QMessageBox.information(self, APP_NAME, TEXT[self.lang]["need_image"])
            return

        stem = Path(self.image_path).stem if self.image_path else "image"
        if self.output_format == "png":
            ext = ".png"
            filt = "PNG (*.png)"
        elif self.output_format == "bmp":
            ext = ".bmp"
            filt = "BMP (*.bmp)"
        else:
            ext = ".jpg"
            filt = "JPEG (*.jpg *.jpeg)"

        if self.image_path:
            default = str(Path(self.image_path).with_name(stem + "_wb" + ext))
        else:
            default = stem + "_wb" + ext

        path, _ = QFileDialog.getSaveFileName(self, APP_NAME, default, filt)
        if not path:
            return

        # The format dropdown is authoritative. Replace an accidentally typed
        # mismatched extension so the filename and actual encoder always agree.
        p = Path(path)
        if self.output_format == "png":
            if p.suffix.lower() != ".png":
                path = str(p.with_suffix(".png"))
        elif self.output_format == "bmp":
            if p.suffix.lower() != ".bmp":
                path = str(p.with_suffix(".bmp"))
        else:
            if p.suffix.lower() not in (".jpg", ".jpeg"):
                path = str(p.with_suffix(".jpg"))

        self.save_image(path)

    def closeEvent(self, event):
        self.settings.sync()
        super().closeEvent(event)

    def save_image(self, path: str):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            out = apply_pipeline(self.full_rgb, self.state.params)
            ext = Path(path).suffix.lower()
            save_kwargs = {}
            if self.icc_profile:
                save_kwargs["icc_profile"] = self.icc_profile
            if self.exif_bytes:
                save_kwargs["exif"] = self.exif_bytes
            if self.dpi:
                save_kwargs["dpi"] = self.dpi

            if ext in (".jpg", ".jpeg"):
                if self.full_alpha is not None:
                    a = self.full_alpha.astype(np.float32)[..., None] / 255.0
                    out_jpg = np.rint(out.astype(np.float32) * a + 255.0 * (1.0 - a)).astype(np.uint8)
                else:
                    out_jpg = out
                img = Image.fromarray(out_jpg, "RGB")
                img.save(path, quality=self.jpeg_quality, subsampling=0, optimize=True, **save_kwargs)
            elif ext == ".png":
                if self.full_alpha is not None:
                    rgba = np.dstack([out, self.full_alpha])
                    img = Image.fromarray(rgba, "RGBA")
                else:
                    img = Image.fromarray(out, "RGB")
                img.save(path, compress_level=self.png_compression, **save_kwargs)
            elif ext == ".bmp":
                # Save BMP as conventional 24-bit RGB for broad compatibility.
                # If the source has alpha, composite it over white because Pillow's
                # BMP writer does not preserve alpha reliably.
                if self.full_alpha is not None:
                    a = self.full_alpha.astype(np.float32)[..., None] / 255.0
                    out_bmp = np.rint(out.astype(np.float32) * a + 255.0 * (1.0 - a)).astype(np.uint8)
                else:
                    out_bmp = out
                img = Image.fromarray(out_bmp, "RGB")
                img.save(path, **save_kwargs)
            else:
                raise ValueError(f"Unsupported output format: {ext}")
            self.status.showMessage(TEXT[self.lang]["save_ok"].format(name=Path(path).name), 8000)
        except Exception as exc:
            QMessageBox.critical(self, APP_NAME, TEXT[self.lang]["save_error"].format(error=str(exc)))
        finally:
            QApplication.restoreOverrideCursor()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

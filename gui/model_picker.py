"""Fetch-models button that turns a model QLineEdit into a picker.

The model field stays a free-text ``QLineEdit`` (so typed names, project
sync and ``_user_customized`` tracking keep working); fetching attaches a
filter-as-you-type completer listing the endpoint's models and opens it.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QStringListModel, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCompleter,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)

from process.custom_endpoint import CustomEndpoint, ModelListError, list_models

EndpointGetter = Callable[[], CustomEndpoint | None]


class ModelFetchButton(QPushButton):
    """Button that lists an endpoint's models into ``target``'s completer."""

    def __init__(
        self,
        target: QLineEdit,
        endpoint_getter: EndpointGetter,
        parent: QWidget | None = None,
    ):
        super().__init__("Fetch Models", parent)
        self._target = target
        self._endpoint_getter = endpoint_getter
        self._model = QStringListModel(self)
        completer = QCompleter(self._model, target)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setMaxVisibleItems(15)
        completer.activated.connect(self._on_model_chosen)
        target.setCompleter(completer)
        self.setAccessibleName("Fetch models from custom endpoint")
        self.setToolTip("List the models served by the custom endpoint")
        self.clicked.connect(self.fetch)

    @property
    def models(self) -> list[str]:
        return self._model.stringList()

    def _on_model_chosen(self, _text: str) -> None:
        # Picking from the list is a user edit, same as typing.
        self._target._user_customized = True

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self.window(), "Fetch Models", message)

    def fetch(self) -> list[str]:
        """Fetch models synchronously and open the picker; returns the IDs."""
        endpoint = self._endpoint_getter()
        if endpoint is None:
            self._show_error("Select the 'custom' provider to fetch its models.")
            return []
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        error = ""
        try:
            models = list_models(endpoint)
        except ModelListError as exc:
            models, error = [], str(exc)
        finally:
            QApplication.restoreOverrideCursor()
        if error:
            self._show_error(error)
            return []
        self._model.setStringList(models)
        completer = self._target.completer()
        self._target.setFocus()
        completer.setCompletionPrefix("")
        completer.complete()
        return models

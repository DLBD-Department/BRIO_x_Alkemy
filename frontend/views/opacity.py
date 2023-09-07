from flask import Flask, render_template, request, redirect, flash, url_for, session, Response, jsonify, Blueprint, current_app

bp = Blueprint('opacity', __name__, template_folder="../templates/bias", url_prefix="/opacity")

@bp.route('/', methods=['GET'])
def opacity_home():
    return "<p>Hello, World!</p>"

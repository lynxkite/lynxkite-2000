<#import "template.ftl" as layout>
<@layout.registrationLayout; section>

    <#if section = "form">
        <div class="login-card" style="text-align: center;">

            <div style="margin-bottom: 1.5rem;">
                <svg width="56" height="56" viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin: 0 auto 1rem; display: block;">
                    <circle cx="28" cy="28" r="27" stroke="rgba(57,188,243,0.3)" stroke-width="2" fill="rgba(57,188,243,0.05)"/>
                    <path d="M20 28h16M28 20l8 8-8 8" stroke="#39bcf3" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>

            <h2 class="login-title" style="margin-bottom: 0.75rem;">Sign out</h2>

            <p style="color: rgba(255,255,255,0.5); font-size: 0.9rem; margin-bottom: 2rem; line-height: 1.6;">
                Are you sure you want to sign out?
            </p>

            <form action="${url.logoutConfirmAction}" method="POST">
                <input type="hidden" name="session_code" value="${logoutConfirm.code}">

                <button type="submit" class="login-button" style="margin-bottom: 1rem;">
                    Yes, sign me out
                </button>
            </form>

            <a href="javascript:history.back()" style="display:block; color: rgba(255,255,255,0.4); font-size: 0.85rem; text-decoration: none; margin-top: 0.75rem; transition: color 0.2s;">
                No, take me back
            </a>

        </div>
    </#if>

</@layout.registrationLayout>
